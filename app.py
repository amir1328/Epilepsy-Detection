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
                    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
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
                full_df = pd.read_csv('epilepsy_federated_dataset.csv')
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

    st.subheader("Input EEG Data")
    st.info("Expects Input Shape: (1, 18, 1024, 1) -> 18 Channels, 1024 Time Steps")
    
    if st.button("Generate Dummy Seizure Data (Test Mode)"):
        # Create a mock signal
        # We will inject a 'seizure' signal (high freq sine wave) into specific channels
        # to see if our localization logic can find it.
        
        # Base: Random noise
        dummy_data = np.random.normal(0, 0.1, (1, 18, 1024, 1)).astype(np.float32)
        
        # Inject "Seizure" into Channel 6 (Temporal Left Area in our list: 'T7-P7' is index 2, let's pick index 6 'C3-P3' Parietal LC)
        # Let's target 'T7-P7' which usually indicates Temporal Lobe epilepsy. Index 2.
        target_channel_idx = 2 
        target_channel_name = channels[target_channel_idx]
        
        # Inject high amplitude sine wave
        t = np.linspace(0, 10, 1024)
        dummy_data[0, target_channel_idx, :, 0] += 5.0 * np.sin(2 * np.pi * 10 * t) # 10Hz signal
        
        st.write(f"**Test Scenario**: Injected strong 10Hz signal into channel **{target_channel_name}**.")
        
        # 1. Run Baseline Prediction
        baseline_pred = dl_model.predict(dummy_data, verbose=0)[0][0]
        st.metric("Baseline Seizure Probability", f"{baseline_pred:.4f}")
        
        # 2. Occlusion Sensitivity (Localization Logic)
        st.write("### Scanning Brain Regions...")
        importances = []
        
        progress_bar = st.progress(0)
        
        for i in range(18):
            # Create occluded copy
            occluded_data = dummy_data.copy()
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
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Channel', y='Importance', data=results, ax=ax, palette='Reds_r')
        plt.xticks(rotation=45)
        ax.set_title("Brain Region Contributions (Occlusion Sensitivity)")
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretation:**
        - The **Higher the bar**, the more critical that brain region is for the seizure.
        - **Stimulation Target**: The region corresponding to the highest bar is the primary focus.
        """)

