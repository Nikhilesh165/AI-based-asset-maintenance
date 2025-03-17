# AI-Driven Transformer Routing with Dynamic Repair Estimation 

This Streamlit web application optimizes asset (Substation-transformer) grid routing by integrating sensor data, asset geometry, and meter information. It leverages an AI model via Ollama to compute rectification steps and complexity scores for failing assets. The application then computes both a default sequential route (based solely on spatial proximity) and a locally rerouted flow that uses a pool of "unused" assets for replacements.

---

## Features

- **File Uploads:**
  - **Sensor Data CSV:** Contains sensor readings with columns such as `LOCATION`, `METER`, `TIMESTAMP`, and `VALUE`.
  - **Asset Geometry CSV:** Provides geographic coordinates (LAT/LON) for each asset (extracted from a `GEOMETRY` field).
  - **Meter Info CSV:** Contains detailed meter information including `GROUPNAME`, `METERNAME`, `METERTYPE`, `DOMAINID`, and `DESCRIPTION`.
  - **Context PDFs:** Technical manuals and repair guides used as contextual data for AI-based rectification queries.

- **Anomaly Classification & Aggregation:**
  - **Classification:** Each sensor record is classified based on its `VALUE`. The application preserves letters (A, B, or C) if available, while mapping "E" to **Emergency** and "D" to **Degraded**.
  - **Aggregation:** Sensor records are grouped by asset (`LOCATION`). The aggregated condition is determined by:
    - **Emergency:** If any record is "Emergency".
    - **Degraded:** Else if any record is "Degraded".
    - **A/B/C:** Otherwise, the mode (most frequent) letter among A, B, or C is used.
  - **Unused Assets:** Assets with an aggregated condition of **B** are flagged as **UNUSED**. These assets are reserved for local rerouting.

- **Asset Complexity Computation:**
  - For each asset, an average complexity score is computed from failing sensor records (Emergency/Degraded) by calling an external AI model.
  - These complexity values are stored with the aggregated asset data and used later to compute route criticality.

- **Default Sequential Route:**
  - A default route is computed using a greedy nearest-neighbor algorithm based solely on spatial proximity (LAT/LON).
  - Only assets that are **not** marked as UNUSED (i.e., those with aggregated condition other than B) are used.

- **Local Replacement Rerouting:**
  - When a failing asset is detected in the default route, the application searches among the UNUSED assets for a nearby replacement.
  - The failing asset is replaced by a candidate healthy asset, and a new sequential route is computed from the union of the default route and these replacements.
  - New route metrics (total Euclidean distance and route criticality defined as the sum of `SEVERITY × COMPLEXITY`) are displayed.

- **Detailed Rectification Steps Display:**
  - Users can select a specific asset and meter via drop-down menus.
  - The application queries the external AI model to display detailed rectification steps and complexity for that asset/meter combination.

- **Interactive Mapping:**
  - Uses Folium (via `streamlit_folium`) to render interactive maps showing the sequential route.
  - Routes are visualized with markers (displaying asset ID, severity, condition, and complexity) and connected by purple polylines.

---
