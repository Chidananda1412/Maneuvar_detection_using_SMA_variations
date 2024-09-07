
# Maneuver Detection Using Semi-Major Axis (SMA) Variations

## Overview

This project aims to detect orbital maneuvers based on variations in the Semi-Major Axis (SMA) over time. Using a heuristic-based approach, the algorithm identifies significant changes in SMA, often indicative of engine burns or orientation adjustments.

The project consists of the following modules:

1. **Data Preprocessing**: Converts and cleans the input data.
2. **Feature Extraction**: Extracts relevant features from the preprocessed data.
3. **Maneuver Detection**: Identifies potential maneuvers using a heuristic approach.
4. **Visualization**: Plots the results to highlight detected maneuvers.

---

## Files

- **SMA_data.csv**: The input data file containing SMA values and corresponding DateTime values.
- **maneuver_detection.py**: The Python script implementing the entire process from data preprocessing to maneuver detection and result visualization.
- **README.md**: This document providing an overview of the project and instructions.

---

## Requirements

This project requires the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

---

## How It Works

1. **Data Preprocessing**: 
   - The SMA data is loaded, and DateTime is converted to the appropriate format.
   - Differences between consecutive SMA values are calculated to detect potential maneuvers.
   
2. **Feature Extraction**:
   - SMA differences are standardized using `StandardScaler` for better anomaly detection.

3. **Maneuver Detection**:
   - Significant changes in SMA differences are flagged.
   - Potential maneuvers are identified by detecting peaks in the SMA data followed by stabilization.

4. **Visualization**:
   - Two plots are generated:
     1. SMA values with detected maneuvers.
     2. SMA differences with detected maneuvers.
   - Detected maneuvers are highlighted in red on the plots.

---

## Usage

1. Clone the repository or download the code.
2. Ensure the **SMA_data.csv** file is placed in the same directory as the Python script.
3. Run the `maneuver_detection.py` file:
   
   ```bash
   python maneuver_detection.py
   ```

   This will preprocess the data, detect maneuvers, and display the resulting visualizations.

---

## Output

The script generates two plots:
1. **SMA vs Detected Maneuvers**: Highlights the detected maneuvers on the SMA data over time.
2. **SMA Difference vs Detected Maneuvers**: Visualizes the SMA differences and highlights the maneuvers.

Additionally, a list of detected maneuver dates and SMA values is printed to the console.



## Methodology

The heuristic approach is chosen for its simplicity and efficiency. It detects potential maneuvers by identifying peaks in SMA values, which typically correspond to engine burns or orientation changes. Stabilization is determined when SMA differences approach zero.

Key Parameters:
- **Threshold**: The minimum change in SMA difference required to flag a potential maneuver.
- **Peak Window**: The window within which a local peak is detected.
- **Stabilization Threshold**: The threshold below which SMA differences indicate stabilization.
- **Stabilization Period**: The number of consecutive points below the stabilization threshold to confirm a stable orbit.

---

## Assumptions

1. SMA variations provide enough information to detect maneuvers.
2. A peak in the SMA data signifies a maneuver, and subsequent stabilization confirms it.
3. The provided dataset is pre-processed and cleaned for use in this project.



