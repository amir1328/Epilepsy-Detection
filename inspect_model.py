import os
import tensorflow as tf

model_path = 'CHB_MIT_sz_detec_demo.h5'

try:
    model = tf.keras.models.load_model(model_path)
    
    print("\n" + "="*30)
    print("CRITICAL MODEL INFO")
    print("="*30)
    for i, input_layer in enumerate(model.inputs):
        print(f"INPUT_{i}_SHAPE: {input_layer.shape}")
        
    for i, output_layer in enumerate(model.outputs):
        print(f"OUTPUT_{i}_SHAPE: {output_layer.shape}")
    print("="*30 + "\n")

except Exception as e:
    print(f"Error: {e}")
