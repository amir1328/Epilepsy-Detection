import pandas as pd
import numpy as np

# Standard channels as defined in app.py
channels = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

# Generate 5 seconds of data at 256 Hz = 1280 samples (we need at least 1024)
n_samples = 1280
data = np.random.normal(0, 1, (n_samples, 18))

# Create DataFrame
df = pd.DataFrame(data, columns=channels)

# Save to CSV
df.to_csv('sample_eeg_input.csv', index=False)
print("Created sample_eeg_input.csv with shape:", df.shape)
