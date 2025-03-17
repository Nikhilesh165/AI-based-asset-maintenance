import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import ollama
import numpy as np
import json
import re
import faiss
import PyPDF2
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------
# Load embedding model (for PDF chunk embeddings)
# ------------------------------------------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------------------------------------------
# FUNCTIONS FOR QUERYING THE OLLAMA API
# ------------------------------------------------------------------------------
def query_ollama(query, context):
    model = "deepseek-r1:7b"
    prompt = f"Context: {context}\n\n{query}"
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# ------------------------------------------------------------------------------
# FUNCTIONS FOR LOADING AND PROCESSING CSV FILES
# ------------------------------------------------------------------------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def classify_anomalies(df):
    """
    For each sensor record, preserve the original letter if it is one of A, B, or C.
    Otherwise, map E -> Emergency, D -> Degraded; if missing then Unknown.
    """
    def determine_condition(row):
        if pd.isna(row["VALUE"]):
            return "Unknown"
        value = str(row["VALUE"]).strip().upper()
        if value == "E":
            return "Emergency"
        elif value == "D":
            return "Degraded"
        elif value in ["A", "B", "C"]:
            return value  # preserve the letter
        return "Unknown"
    df["CONDITION"] = df.apply(determine_condition, axis=1)
    return df

def aggregate_asset_condition(sensor_data):
    """
    Aggregate sensor data by asset (LOCATION).
      - If any record is "Emergency" then aggregated condition is "Emergency".
      - Else if any record is "Degraded" then aggregated condition is "Degraded".
      - Else (all records in {A, B, C}) use the mode letter.
    Also, compute a severity value: Emergency=3, Degraded=2, and for A, B, or C use 1.
    """
    groups = sensor_data.groupby("LOCATION")
    agg_list = []
    for loc, group in groups:
        if "Emergency" in group["CONDITION"].values:
            agg_cond = "Emergency"
        elif "Degraded" in group["CONDITION"].values:
            agg_cond = "Degraded"
        else:
            mode_val = group[group["CONDITION"].isin(["A","B","C"])]["CONDITION"].mode()
            if not mode_val.empty:
                agg_cond = mode_val.iloc[0]
            else:
                agg_cond = "Unknown"
        first_row = group.iloc[0]
        severity = 3 if agg_cond=="Emergency" else 2 if agg_cond=="Degraded" else 1 if agg_cond in ["A","B","C"] else 0
        agg_list.append({
            "LOCATION": loc,
            "AGG_CONDITION": agg_cond,
            "LAT": first_row["LAT"],
            "LON": first_row["LON"],
            "SEVERITY": severity
        })
    agg_df = pd.DataFrame(agg_list)
    # Mark assets with aggregated condition "B" as UNUSED (reserved for rerouting)
    agg_df["UNUSED"] = agg_df["AGG_CONDITION"].apply(lambda x: True if x=="B" else False)
    return agg_df

# ------------------------------------------------------------------------------
# FUNCTIONS FOR PROCESSING PDF CONTEXT (FOR TECHNICAL MANUALS, ETC.)
# ------------------------------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start:start+chunk_size]
        chunks.append(" ".join(chunk))
        start += (chunk_size - overlap)
    return chunks

def process_pdfs(pdf_files):
    all_chunks = []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    if all_chunks:
        embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
    else:
        index = None
    return all_chunks, index

def get_relevant_pdf_context(query, index, chunks, top_k=3):
    if index is None or not chunks:
        return ""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return "\n".join(relevant_chunks)

# ------------------------------------------------------------------------------
# FUNCTION FOR ANOMALY RECTIFICATION (FOR DETAIL DISPLAY)
# ------------------------------------------------------------------------------
def get_rectification_info_for_anomaly(anomaly, faiss_index, pdf_chunks):
    """
    Query the external model for rectification steps and a complexity score for a sensor record.
    Expects JSON output with a "rectification_steps" key, where each step has a "complexity" (1-10).
    Returns (rectification_steps, average_complexity, severity).
    """
    severity_order = {"Emergency": 3, "Degraded": 2, "A": 1, "B": 1, "C": 1, "Unknown": 0}
    severity = severity_order.get(anomaly["CONDITION"], 0)
    pdf_query = (
        f"Rectification steps for an anomaly at location {anomaly['LOCATION']} on meter {anomaly['METER']} "
        f"with condition {anomaly['CONDITION']} and value {anomaly['VALUE']}."
    )
    pdf_context = get_relevant_pdf_context(pdf_query, faiss_index, pdf_chunks, top_k=3)
    meter_description = anomaly.get("DESCRIPTION", "N/A")
    meter_type = anomaly.get("METERTYPE", "N/A")
    prompt = (
        f"Using the following context from technical manuals and repair guides:\n{pdf_context}\n\n"
        f"Provide the rectification steps for the anomaly with the following details:\n"
        f"- Location: {anomaly['LOCATION']}\n"
        f"- Meter: {anomaly['METER']} (Description: {meter_description}, Type: {meter_type})\n"
        f"- Condition: {anomaly['CONDITION']}\n"
        f"- Value: {anomaly['VALUE']}\n"
        f"- Calculated Severity: {severity}\n\n"
        "Return the answer in JSON format with a key 'rectification_steps' that is a list of steps. "
        "Each step should be an object with keys 'step' and 'complexity' (numeric between 1 and 10). "
        "Only provide the JSON output."
    )
    response_text = query_ollama(prompt, "Rectification based on PDF context for individual anomaly")
    try:
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
            info = json.loads(json_text)
        else:
            info = json.loads(response_text)
        rectification_steps = info.get("rectification_steps", [])
        if isinstance(rectification_steps, list) and rectification_steps:
            complexities = []
            for step in rectification_steps:
                try:
                    comp = float(step.get("complexity", 5))
                    complexities.append(comp)
                except Exception:
                    complexities.append(5.0)
            final_complexity = sum(complexities) / len(complexities) if complexities else 5.0
        else:
            final_complexity = float(info.get("complexity", 5))
        return rectification_steps, final_complexity, severity
    except Exception as e:
        st.error(f"Error parsing model response: {e}; using default complexity of 5.\nModel response: {response_text}")
        return response_text, 5.0, severity

# ------------------------------------------------------------------------------
# FUNCTIONS FOR COMPUTING SEQUENTIAL ROUTES
# ------------------------------------------------------------------------------
def compute_sequential_route_default(assets_df):
    """
    Compute a sequential route (using a greedy nearest-neighbor algorithm) based solely on spatial proximity.
    Expects to be given only the default assets (i.e. those NOT marked as UNUSED).
    Returns the ordered DataFrame and total Euclidean distance.
    """
    if assets_df.empty:
        return assets_df, 0
    remaining = assets_df.copy()
    initial_idx = remaining['LAT'].idxmin()
    route_indices = [initial_idx]
    remaining = remaining.drop(index=initial_idx)
    while not remaining.empty:
        last_asset = assets_df.loc[route_indices[-1]]
        distances = np.sqrt((remaining['LAT'] - last_asset['LAT'])**2 + (remaining['LON'] - last_asset['LON'])**2)
        next_idx = distances.idxmin()
        route_indices.append(next_idx)
        remaining = remaining.drop(index=next_idx)
    ordered_df = assets_df.loc[route_indices]
    total_distance = 0
    for i in range(len(ordered_df) - 1):
        lat1, lon1 = ordered_df.iloc[i]['LAT'], ordered_df.iloc[i]['LON']
        lat2, lon2 = ordered_df.iloc[i+1]['LAT'], ordered_df.iloc[i+1]['LON']
        total_distance += np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    return ordered_df, total_distance

def compute_route_metrics(ordered_df):
    """
    Compute route criticality as the sum over the route of (SEVERITY × COMPLEXITY) and also total distance.
    """
    total_distance = 0
    total_criticality = 0
    for i in range(len(ordered_df)):
        row = ordered_df.iloc[i]
        total_criticality += row["SEVERITY"] * row.get("COMPLEXITY", 0)
        if i < len(ordered_df) - 1:
            lat1, lon1 = row['LAT'], row['LON']
            lat2, lon2 = ordered_df.iloc[i+1]['LAT'], ordered_df.iloc[i+1]['LON']
            total_distance += np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    return total_distance, total_criticality

def render_sequential_route(ordered_df, total_distance, extra_metric=None, title="Sequential Route"):
    if ordered_df.empty:
        st.error("No assets to display in the route.")
        return
    initial_lat = ordered_df.iloc[0]['LAT']
    initial_lon = ordered_df.iloc[0]['LON']
    folium_map = folium.Map(location=[initial_lat, initial_lon], zoom_start=12)
    coordinates = []
    for idx, row in ordered_df.iterrows():
        lat = row['LAT']
        lon = row['LON']
        coordinates.append((lat, lon))
        marker_text = f"{row['LOCATION']} (Severity: {row['SEVERITY']}, Condition: {row.get('AGG_CONDITION','')}, Complexity: {row.get('COMPLEXITY','N/A')})"
        folium.Marker(location=[lat, lon], popup=marker_text, icon=folium.Icon(color="blue")).add_to(folium_map)
    folium.PolyLine(coordinates, color="purple", weight=3, opacity=0.8).add_to(folium_map)
    st.write(f"**{title}**: Total Distance = {total_distance:.2f}")
    if extra_metric is not None:
        st.write(f"**Route Criticality** (sum of SEVERITY × COMPLEXITY) = {extra_metric:.2f}")
    folium_static(folium_map)

# ------------------------------------------------------------------------------
# NEW FUNCTION: COMPUTE ROUTE WITH LOCAL REPLACEMENTS
# ------------------------------------------------------------------------------
def compute_route_with_replacements(aggregated_df, default_route_df):
    """
    Given the default route (computed from assets not marked as UNUSED), for each failing asset in that route,
    search among assets that are marked as UNUSED (i.e. aggregated condition "B") and not already in the default route.
    If a candidate is found (the nearest healthy candidate), add it as a replacement.
    Then compute a new sequential route from the union of the default route and these replacements.
    """
    used_indices = list(default_route_df.index)
    replacement_indices = []
    for idx in default_route_df.index:
        asset = default_route_df.loc[idx]
        if asset["CONDITION"] in ["Emergency", "Degraded"]:
            # Look for replacement candidates among assets marked as UNUSED
            candidates = aggregated_df[(aggregated_df["UNUSED"] == True) & (~aggregated_df.index.isin(used_indices))]
            if not candidates.empty:
                distances = np.sqrt((candidates["LAT"] - asset["LAT"])**2 + (candidates["LON"] - asset["LON"])**2)
                candidate_idx = distances.idxmin()
                replacement_indices.append(candidate_idx)
                used_indices.append(candidate_idx)
                st.write(f"Replacing failing asset {asset['LOCATION']} with candidate {aggregated_df.loc[candidate_idx]['LOCATION']} (distance {distances.loc[candidate_idx]:.2f})")
            else:
                st.write(f"No replacement found for failing asset {asset['LOCATION']}.")
    if replacement_indices:
        new_set = pd.concat([default_route_df, aggregated_df.loc[replacement_indices]])
    else:
        new_set = default_route_df
    new_order, new_total_distance = compute_sequential_route_default(new_set)
    return new_order, new_total_distance

# ------------------------------------------------------------------------------
# FUNCTION TO DISPLAY RECTIFICATION STEPS FOR A CHOSEN ASSET & METER
# ------------------------------------------------------------------------------
def display_rectification_steps_for_asset_meter(sensor_data, faiss_index, pdf_chunks):
    assets = sensor_data['LOCATION'].unique().tolist()
    selected_asset = st.selectbox("Select Asset", assets, key="rect_asset")
    asset_rows = sensor_data[sensor_data["LOCATION"] == selected_asset]
    meters = asset_rows['METER'].unique().tolist()
    selected_meter = st.selectbox("Select Meter", meters, key="rect_meter")
    row = asset_rows[asset_rows["METER"] == selected_meter].iloc[0]
    if st.button("Show Rectification Steps for Selected Asset and Meter", key="show_rect_steps"):
        rectification_steps, comp, sev = get_rectification_info_for_anomaly(row, faiss_index, pdf_chunks)
        st.write("### Rectification Steps:")
        st.write(rectification_steps)
        st.write("**Average Complexity:**", comp)
        st.write("**Meter Severity:**", sev)

# ------------------------------------------------------------------------------
# MAIN STREAMLIT APPLICATION
# ------------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("AI-Driven Transformer Routing with Sequential Flow Rerouting")

# Sidebar: Upload Files
st.sidebar.header("📜 Upload Files")
uploaded_sensor_csv = st.sidebar.file_uploader("Upload Sensor Data CSV", type=["csv"])
uploaded_geo_csv = st.sidebar.file_uploader("Upload Asset Geometry CSV", type=["csv"])
uploaded_meter_csv = st.sidebar.file_uploader("Upload Meter Info CSV", type=["csv"])
uploaded_context_pdfs = st.sidebar.file_uploader("Upload Context PDFs", type=["pdf"], accept_multiple_files=True)

# Process PDFs if provided.
if uploaded_context_pdfs:
    pdf_chunks, faiss_index = process_pdfs(uploaded_context_pdfs)
else:
    pdf_chunks = []
    faiss_index = None

if uploaded_sensor_csv and uploaded_geo_csv and uploaded_meter_csv:
    # Process sensor data.
    sensor_data = load_csv(uploaded_sensor_csv)
    sensor_data["VALUE"] = pd.to_numeric(sensor_data["VALUE"], errors="coerce")
    sensor_data = classify_anomalies(sensor_data)
    
    # Load asset geometry and extract coordinates.
    geo_df = load_csv(uploaded_geo_csv)
    geo_df[['LON', 'LAT']] = geo_df['GEOMETRY'].str.extract(r"POINT \(([-\d\.]+) ([-\d\.]+)\)").astype(float)
    
    # Load meter info.
    meter_info = load_csv(uploaded_meter_csv)
    
    # Merge sensor data with geometry and meter info.
    sensor_data = sensor_data.merge(geo_df[['LOCATION', 'LAT', 'LON']], on="LOCATION", how="left")
    sensor_data = sensor_data.merge(meter_info, left_on="METER", right_on="METERNAME", how="left")
    
    # Aggregate sensor data by asset.
    aggregated_data = aggregate_asset_condition(sensor_data)
    
    # Compute asset complexities (precomputed) and add them.
    asset_complexities = {} 
    # Call compute_asset_complexities over sensor_data for each asset in aggregated_data.
    for loc in aggregated_data["LOCATION"]:
        subset = sensor_data[sensor_data["LOCATION"] == loc]
        failing = subset[subset["CONDITION"].isin(["Emergency", "Degraded"])]
        complexities = []
        for _, row in failing.iterrows():
            _, comp, _ = get_rectification_info_for_anomaly(row, faiss_index, pdf_chunks)
            complexities.append(comp)
        asset_complexities[loc] = (sum(complexities)/len(complexities)) if complexities else 0
    aggregated_data["COMPLEXITY"] = aggregated_data["LOCATION"].map(asset_complexities)
    
    # Partition aggregated assets into two groups:
    # - Default assets: those NOT marked as UNUSED (i.e. aggregated condition != "B")
    # - Unused assets: those marked as UNUSED (i.e. aggregated condition == "B")
    default_assets = aggregated_data[aggregated_data["UNUSED"] == False]
    
    st.subheader("Default Sequential Route (Spatial Only)")
    default_order, default_distance = compute_sequential_route_default(default_assets)
    render_sequential_route(default_order, default_distance, title="Default Sequential Route")
    
    st.markdown("---")
    st.subheader("Recompute Route with Local Replacement of Failing Assets")
    if st.button("Recompute Route with Local Replacements", key="reroute_local"):
        new_order, new_total_distance = compute_route_with_replacements(aggregated_data, default_order)
        new_distance, new_criticality = compute_route_metrics(new_order)
        st.write(f"**New Route Metrics:** Total Distance = {new_distance:.2f}, Route Criticality = {new_criticality:.2f}")
        render_sequential_route(new_order, new_distance, extra_metric=new_criticality, title="Rerouted Sequential Route (Local Replacements)")
    
    st.markdown("---")
    st.subheader("Display Rectification Steps for a Specific Asset & Meter")
    display_rectification_steps_for_asset_meter(sensor_data, faiss_index, pdf_chunks)
