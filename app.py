import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

# Load Model and Scaler
@st.cache_resource
def load_all_models():
    # Load original RF model
    rf_model = joblib.load('epilepsy_rf_model.pkl')
    scaler = joblib.load('epilepsy_scaler.pkl')
    
    # Load CHB-MIT Deep Learning Model
    try:
        dl_model = tf.keras.models.load_model('CHB_MIT_sz_detec_demo.h5')
    except:
        dl_model = None
        
    return rf_model, scaler, dl_model

st.title("🧠 Epilepsy Prediction & Localization")
st.markdown("Dual-system: **Seizure Classification** (Random Forest) and **Brain Localization** (CHB-MIT Deep Learning).")

try:
    rf_model, scaler, dl_model = load_all_models()
    st.success("Models loaded successfully!")
    if dl_model is None:
        st.warning("CHB-MIT model not found. Localization feature disabled.")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Information about classes
classes = {
    0: "Class 0 (Non-Seizure / Normal)",
    1: "Class 1 (Seizure Type 1)",
    2: "Class 2 (Seizure Type 2)"
}

# Sidebar
st.sidebar.header("Mode Selection")
app_mode = st.sidebar.radio("Choose Task:", ["Seizure Classification (CSV)", "Brain Localization (High-Density)"])

if app_mode == "Seizure Classification (CSV)":
    st.header("1. Seizure Type Classification")
    st.markdown("Predict seizure type (Class 0, 1, 2) from tabular EEG features.")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV Features", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.write(input_df.head())

        # Preprocessing check
        # Ensure columns match (basic check)
        # Note: In a real app we'd need rigorous checking. here we assume user uploads valid feature set.
        
        # We need to drop label columns if they exist in the input, so we don't accidentally use them as features or cause scaler errors
        cols_to_drop = ['Seizure_Type_Label', 'Multi_Class_Label']
        features_df = input_df.drop(columns=[c for c in cols_to_drop if c in input_df.columns], errors='ignore')
        
        if st.button("Predict"):
            try:
                # Scale
                features_scaled = scaler.transform(features_df)
                
                # Predict
                predictions = rf_model.predict(features_scaled)
                probs = rf_model.predict_proba(features_scaled)
                
                # Results
                input_df['Prediction'] = predictions
                input_df['Prediction_Label'] = input_df['Prediction'].map(classes)
                input_df['Confidence'] = np.max(probs, axis=1)
                
                st.write("### Prediction Results")
                st.dataframe(input_df[['Prediction', 'Prediction_Label', 'Confidence']].head(10))
                
                # Feature Importance
                st.write("### Usage of Features (Model Interpretability)")
                st.info("Since this dataset does not contain spatial data (electrode locations), we cannot map activity to specific brain regions. However, we can show which signal characteristics were most important for this prediction.")
                
                if hasattr(rf_model, 'feature_importances_'):
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    # Get feature importance
                    importances = rf_model.feature_importances_
                    feature_names = features_df.columns
                    
                    # Create DataFrame for plotting
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False).head(10) # Top 10
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, hue='Feature', palette='viridis', legend=False)
                    ax.set_title('Top 10 Features Driving Prediction')
                    ax.set_xlabel('Importance Score')
                    ax.set_ylabel('EEG Feature')
                    st.pyplot(fig)
                else:
                    st.warning("This model type does not support feature importance extraction.")

                # Summary
                st.write("### Summary Stats")
                st.bar_chart(input_df['Prediction_Label'].value_counts())
                
                # Download
                csv = input_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Predictions",
                    csv,
                    "epilepsy_predictions.csv",
                    "text/csv",
                    key='download-csv'
                )
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.error("Please ensure your input CSV has the correct feature columns.")

    elif uploaded_file is None:
        st.info("Awaiting CSV file upload.")
        
        if st.button("Load Sample Data (Random 5 rows)"):
             # ... (existing sample logic, minimal change needed here just indentation)
             # Repasting existing logic with correct indentation for the 'if' block
            try:
                full_df = pd.read_csv('sample_input.csv')
                sample = full_df.sample(5)
                # ... process sample
                features_df = sample.drop(columns=['Seizure_Type_Label', 'Multi_Class_Label'], errors='ignore')
                features_scaled = scaler.transform(features_df)
                predictions = rf_model.predict(features_scaled) # Changed to rf_model
                probs = rf_model.predict_proba(features_scaled)
                
                sample['Prediction'] = predictions
                sample['Prediction_Label'] = sample['Prediction'].map(classes)
                sample['Confidence'] = np.max(probs, axis=1)
                
                st.write("### Predictions on Sample")
                st.dataframe(sample[['Seizure_Type_Label', 'Prediction', 'Prediction_Label', 'Confidence']])
            except Exception as e:
                st.error(f"Error: {e}")

elif app_mode == "Brain Localization (High-Density)":
    st.header("2. Brain Localization Map")
    st.markdown("Identify the **Brain Region** responsible for the seizure using the CHB-MIT Deep Learning model.")
    
    if dl_model is None:
        st.error("CHB-MIT Model file is missing.")
        st.stop()

    # Define Standard Channels (A common 18-channel clinical montage)
    # We assume the model follows a standard indexing. 
    # If the user provides specific channel names, we can map them.
    channels = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ'
    ]
    
    # Map channels to Lobes for easy interpretation
    lobe_mapping = {
        'FP1': 'Frontal Left', 'F7': 'Frontal Left', 'F3': 'Frontal Left',
        'FP2': 'Frontal Right', 'F8': 'Frontal Right', 'F4': 'Frontal Right',
        'T7': 'Temporal Left', 'P7': 'Parietal Left', 
        'T8': 'Temporal Right', 'P8': 'Parietal Right',
        'O1': 'Occipital Left', 'O2': 'Occipital Right',
        'FZ': 'Central/Midline', 'CZ': 'Central/Midline', 'PZ': 'Central/Midline',
        'C3': 'Central Left', 'C4': 'Central Right',
        'P3': 'Parietal Left', 'P4': 'Parietal Right'
    }

    # --- Heatmap Logic ---
    from scipy.interpolate import griddata
    import matplotlib.image as mpimg

    def plot_brain_heatmap(channels, importances):
        """
        Plots a heatmap of channel importances on a brain image.
        """
        # 1. Define Channel Coordinates (Normalized 0.0 to 1.0)
        # Assuming image has Top=0, Bottom=1, Left=0, Right=1
        # Frontal is Top (Low Y), Occipital is Bottom (High Y)
        coords = {
            'FP1': (0.37, 0.18), 'FP2': (0.63, 0.18),
            'F7': (0.18, 0.32), 'F3': (0.33, 0.32), 'FZ': (0.5, 0.32), 'F4': (0.67, 0.32), 'F8': (0.82, 0.32),
            'T7': (0.13, 0.50), 'C3': (0.33, 0.50), 'CZ': (0.5, 0.50), 'C4': (0.67, 0.50), 'T8': (0.87, 0.50),
            'P7': (0.18, 0.68), 'P3': (0.33, 0.68), 'PZ': (0.5, 0.68), 'P4': (0.67, 0.68), 'P8': (0.82, 0.68),
            'O1': (0.37, 0.85), 'O2': (0.63, 0.85)
        }

        # Extract x, y, z (importance)
        x_vals = []
        y_vals = []
        z_vals = []
        
        # We process 'channels' input (e.g. 'FP1-F7') to map to single electrode locations
        # Heuristic: Assign the importance of a differential channel 'A-B' to BOTH A and B equally?
        # Or just accumulate.
        
        node_importance = {k: 0.0 for k in coords.keys()}
        node_counts = {k: 0 for k in coords.keys()}
        
        for ch_name, imp in zip(channels, importances):
            # ch_name might be 'FP1-F7'. Split it.
            parts = ch_name.split('-')
            for part in parts:
                if part in node_importance:
                    node_importance[part] += imp
                    node_counts[part] += 1
        
        # Average importance per node
        for node in node_importance:
            if node_counts[node] > 0:
                node_importance[node] /= node_counts[node]
                
        # Prepare points for interpolation
        for node, val in node_importance.items():
            if node in coords:
                x, y = coords[node]
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(val)
                
        # 2. Interpolate to Grid
        grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
        grid_z = griddata((x_vals, y_vals), z_vals, (grid_x, grid_y), method='cubic', fill_value=0)
        
        # 3. Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Load Image
        try:
            # Use PIL for better image support
            from PIL import Image
            img_path = 'brain_outline.png'
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img, extent=[0, 1, 1, 0])
            else:
                st.warning("Brain outline image not found. Using simple axes.")
                ax.invert_yaxis()
        except Exception as e:
            st.error(f"Image load failed: {e}")
            ax.invert_yaxis()
            
        # Overlay Heatmap
        im = ax.imshow(grid_z.T, extent=[0, 1, 0, 1], origin='upper', cmap='jet', alpha=0.5, vmin=0)
        
        # Plot Scatter Points
        ax.scatter(x_vals, y_vals, c='black', s=20, alpha=0.7)
        for i, node in enumerate(node_importance.keys()):
             if node in coords:
                ax.text(coords[node][0], coords[node][1], node, fontsize=8, ha='center', va='bottom', color='black')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Contribution")
        ax.set_title("Seizure Source Localization (Heatmap)")
        ax.axis('off')
        
        return fig

    st.subheader("Input EEG Data")
    st.info("Expects Input Shape: (1, 18, 1024, 1) -> 18 Channels, 1024 Time Steps")
    
    st.subheader("Input EEG Data")
    st.info("Expects Input Shape: (1, 18, 1024, 1) -> 18 Channels (Rows), 1024 Time Steps (Cols)")

    # Data Source Selection
    data_source = st.radio("Select Data Source:", ["Upload CSV", "Generate Dummy Data (Test Mode)"])
    
    eeg_data = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload EEG CSV", type=["csv"])
        if uploaded_file is not None:
             try:
                # Load CSV
                df = pd.read_csv(uploaded_file)
                
                # Check dimensions
                # We expect 18 columns (channels) or 18 rows depending on format. 
                # Let's assume standard format often has Channels as Columns or Time as Columns?
                # The prompt says "18 Channels (Rows), 1024 Time Steps (Cols)" which implies 18x1024.
                # However, Pandas reads into (Rows, Cols). 
                # If the file is (1024, 18) -> 1024 time steps, 18 channels. This is common time-series format.
                # If the file is (18, 1024) -> 18 channels, 1024 time steps.
                
                # Let's support (1024, 18) as standard time-series.
                if df.shape[0] >= 1024 and df.shape[1] >= 18:
                    # Case 1: Time x Channels
                     # Check if columns match our channels list
                    matched_cols = [c for c in channels if c in df.columns]
                    
                    if len(matched_cols) == 18:
                        # Perfect match
                        data_slice = df[channels].iloc[:1024].values # (1024, 18)
                        # We need (1, 18, 1024, 1)
                        # Transpose to (18, 1024)
                        data_slice = data_slice.T
                        # Reshape
                        eeg_data = data_slice.reshape(1, 18, 1024, 1)
                        st.success("Successfully loaded and reshaped data from (Time x Channels) format.")
                        
                    else:
                        # Fallback: Just take first 18 numeric columns
                        numeric_df = df.select_dtypes(include=[np.number])
                        if numeric_df.shape[1] >= 18:
                            data_slice = numeric_df.iloc[:1024, :18].values # (1024, 18)
                            data_slice = data_slice.T # (18, 1024)
                            eeg_data = data_slice.reshape(1, 18, 1024, 1)
                            st.warning(f"Column names didn't match standard channels. Used first 18 numeric columns.")
                        else:
                            st.error("Not enough numeric columns found.")
                            
                elif df.shape[0] >= 18 and df.shape[1] >= 1024:
                    # Case 2: Channels x Time
                    # Take first 18 rows and first 1024 cols
                    data_slice = df.iloc[:18, :1024].values
                    eeg_data = data_slice.reshape(1, 18, 1024, 1)
                    st.success("Successfully loaded data from (Channels x Time) format.")
                    
                else:
                    st.error(f"Invalid Data Shape: {df.shape}. Need at least (1024, 18) or (18, 1024).")

             except Exception as e:
                 st.error(f"Error reading file: {e}")

    elif data_source == "Generate Dummy Data (Test Mode)":
        if st.button("Generate & Run"):
            # Base: Random noise
            dummy_data = np.random.normal(0, 0.1, (1, 18, 1024, 1)).astype(np.float32)
            
            # Inject "Seizure" into Channel 6 (Temporal Left Area in our list: 'T7-P7' is index 2. 
            # Actually 'C3-P3' is index 6. Let's stick to using 'T7-P7' at index 2 for Temporal Left.
            target_channel_idx = 2 
            target_channel_name = channels[target_channel_idx]
            
            # Inject high amplitude sine wave
            t = np.linspace(0, 10, 1024)
            dummy_data[0, target_channel_idx, :, 0] += 5.0 * np.sin(2 * np.pi * 10 * t) # 10Hz signal
            
            st.write(f"**Test Scenario**: Injected strong 10Hz signal into channel **{target_channel_name}**.")
            eeg_data = dummy_data

    # --- PROCESSING ---
    if eeg_data is not None:
        # 1. Run Baseline Prediction
        baseline_pred = dl_model.predict(eeg_data, verbose=0)[0][0]
        st.metric("Baseline Seizure Probability", f"{baseline_pred:.4f}")
        
        # 2. Occlusion Sensitivity (Localization Logic)
        with st.spinner("Scanning Brain Regions..."):
            importances = []
            progress_bar = st.progress(0)
            
            # We need to loop over channels. eeg_data is (1, 18, 1024, 1)
            # Channel axis is 1.
            
            for i in range(18):
                # Create occluded copy
                occluded_data = eeg_data.copy()
                # Zero out the i-th channel
                occluded_data[0, i, :, 0] = 0.0 
                
                # Predict
                pred = dl_model.predict(occluded_data, verbose=0)[0][0]
                
                # Importance = How much did the probability DROP?
                # If drop is high, that channel was critical.
                importance = max(0, baseline_pred - pred)
                importances.append(importance)
                progress_bar.progress((i + 1) / 18)
            
        # 3. Visualize
        results = pd.DataFrame({
            'Channel': channels,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Top culprit
        top_channel = results.iloc[0]
        st.success(f"**Localization Result**: Seizure activity detected in **{top_channel['Channel']}**")
        
        # --- NEW Heatmap Visualization ---
        st.write("### Brain Activity Heatmap")
        fig_heatmap = plot_brain_heatmap(channels, importances)
        st.pyplot(fig_heatmap)
        
        # Original Bar Plot (Optional, keep for detail)
        with st.expander("Show Detailed Bar Chart"):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Channel', y='Importance', data=results, ax=ax, hue='Channel', palette='Reds_r', legend=False)
            plt.xticks(rotation=45)
            ax.set_title("Brain Region Contributions (Occlusion Sensitivity)")
            st.pyplot(fig)
        
        st.markdown("""
        **Interpretation:**
        - The **Red/Hot areas** on the brain map indicate the localized source of the seizure.
        - **Bar Chart** shows raw values for each channel pair.
        """)

