# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

# Load the data
data = pd.read_csv('SMA_data.csv')

# Data Preprocessing Module
def preprocess_data(data):
    # Convert DateTime to pandas datetime object
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    # Calculate the difference in SMA to detect potential maneuvers
    data['SMA_diff'] = data['SMA'].diff().fillna(0)
    return data

# Feature Extraction Module
def extract_features(data):
    # Standardize the SMA difference for better analysis
    scaler = StandardScaler()
    data['SMA_diff_scaled'] = scaler.fit_transform(data[['SMA_diff']])
    return data

# Maneuver Detection Module (Heuristic-Based)
def detect_maneuvers(data, threshold=0.2, peak_window=10, stabilization_threshold=0.2, stabilization_period=5):
    # Detect significant deviations in SMA difference
    data['significant_change'] = np.where(np.abs(data['SMA_diff']) > threshold, 1, 0)
    
    # Create a flag for when SMA_diff is close to zero (indicating stabilization)
    data['stabilized'] = np.where(np.abs(data['SMA_diff']) < stabilization_threshold, 1, 0)
    
    # Detect maneuvers: A maneuver occurs when there's a significant change followed by stabilization
    maneuver_indices = []
    i = 0
    while i < len(data) - peak_window:
        if data['significant_change'].iloc[i] == 1:
            # After detecting a significant change, find the local peak within the window
            peak_index = i + np.argmax(data['SMA'].iloc[i:i + peak_window].values)
            # Check if the peak is actually a local maximum
            if peak_index not in maneuver_indices:
                maneuver_indices.append(peak_index)
            # Move index to the end of the window to avoid repeated detection
            i += peak_window
        else:
            i += 1

    # Mark the detected maneuvers
    data['maneuver_detected'] = 0
    data.loc[maneuver_indices, 'maneuver_detected'] = 1
    
    return data, maneuver_indices

# Result Visualization Module
def visualize_results(data, maneuver_indices):
    # Plot the results: SMA data (in kilometers) vs. detected maneuvers
    plt.figure(figsize=(12, 6))
    plt.plot(data['Datetime'], data['SMA'], label='SMA (km)')

    # Highlight the detected maneuvers on the plot
    detected_maneuvers = data.iloc[maneuver_indices]
    plt.scatter(detected_maneuvers['Datetime'], detected_maneuvers['SMA'], color='red', label='Detected Maneuvers', zorder=5)

    plt.xlabel('Date')
    plt.ylabel('SMA (km)')
    plt.title('SMA (km) vs Detected Maneuvers (Heuristic-Based Detection)')
    plt.legend()
    plt.show()

    # Additional plot to visualize SMA differences
    plt.figure(figsize=(12, 6))
    plt.plot(data['Datetime'], data['SMA_diff'], label='SMA Difference')
    plt.scatter(detected_maneuvers['Datetime'], detected_maneuvers['SMA_diff'], color='red', label='Detected Maneuvers')

    plt.xlabel('Date')
    plt.ylabel('SMA Difference (km)')
    plt.title('SMA Difference vs Detected Maneuvers (Heuristic-Based Detection)')
    plt.legend()
    plt.show()

    # Print detected maneuver dates
    print("Detected Maneuvers:")
    print(detected_maneuvers[['Datetime', 'SMA']])

# Main Function
def main():
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Extract features from the data
    feature_data = extract_features(processed_data)
    
    # Detect maneuvers using the heuristic-based approach
    detected_data, maneuver_indices = detect_maneuvers(feature_data)
    
    # Visualize the results
    visualize_results(detected_data, maneuver_indices)

# Run the main function
if __name__ == "__main__":
    main()
