import os
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from model import build_fusion_resnet
import glob

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
FIGURE_DIR = os.path.join(BASE_DIR, "Code", "figures")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    X_fhr = np.load(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"))
    X_tabular = np.load(os.path.join(PROCESSED_DATA_DIR, "X_tabular.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y.npy"))
    if X_fhr.ndim == 2:
        X_fhr = np.expand_dims(X_fhr, axis=-1)
    return X_fhr, X_tabular, y

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # img_array structure: list of inputs [ts_input, tab_input]?
    # No, Grad-CAM is wrt to the Conv branch input.
    # The model input is a list.
    
    # Create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=[0, 1]) # Axis 0 is batch, 1 is time? 
    # grads shape: (Batch, Time, Filters). Reduce over Time (axis 1).
    # wait, tf.reduce_mean(grads, axis=(0, 1)) results in scalar?
    # We want pooled grads per filter.
    # grads: (1, 3600/stri, 128).
    # We pool over time dimension (axis 1).
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if 'conv1d' in layer.name:
            return layer.name
    return None

def main():
    ensure_dir(FIGURE_DIR)
    X_fhr, X_tabular, y = load_data()
    
    # Load separate models or just one?
    # Let's pick the best model from Fold 1
    model_path = os.path.join(MODEL_DIR, "best_model_fold_1.h5")
    if not os.path.exists(model_path):
        print("Model not found. Run training first.")
        return
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # ------------------ Grad-CAM ------------------
    # Pick a compromised case (Positive)
    pos_indices = np.where(y == 1)[0]
    if len(pos_indices) > 0:
        idx = pos_indices[0]
        input_ts = X_fhr[idx:idx+1]
        input_tab = X_tabular[idx:idx+1]
        
        last_conv_layer = find_last_conv_layer(model)
        if last_conv_layer:
            print(f"Generating Grad-CAM using layer: {last_conv_layer}")
            heatmap = make_gradcam_heatmap([input_ts, input_tab], model, last_conv_layer, pred_index=0)
            
            # Upsample heatmap to original signal length (3600)
            import scipy.ndimage
            zoom_factor = X_fhr.shape[1] / len(heatmap)
            heatmap_resized = scipy.ndimage.zoom(heatmap, zoom_factor, order=1)
            
            plt.figure(figsize=(12, 4))
            plt.plot(input_ts[0], label='FHR')
            plt.imshow([heatmap_resized], aspect='auto', cmap='jet', alpha=0.5, extent=[0, X_fhr.shape[1], 0, 1])
            plt.title(f"Grad-CAM (True Label: {y[idx]})")
            plt.colorbar()
            plt.savefig(os.path.join(FIGURE_DIR, "grad_cam.png"))
            print("Saved grad_cam.png")
        else:
            print("No conv layer found.")
            
    # ------------------ SHAP ------------------
    # Use background data (e.g., 100 samples)
    background_ts = X_fhr[:100]
    background_tab = X_tabular[:100]
    
    # Explain predictions
    # shap.DeepExplainer needs inputs to be list if model takes list
    # But DeepExplainer sometimes struggles with Keras Functional API in TF2.
    # We usually pass the model outputing just the branch we want or the full model?
    # User says "feature importance for the Tabular data".
    # SHAP will return shap values for *all* inputs.
    
    print("Running SHAP...")
    try:
        explainer = shap.DeepExplainer(model, [background_ts, background_tab])
        # Explain next 10 samples
        shap_values = explainer.shap_values([X_fhr[100:110], X_tabular[100:110]])
        
        # shap_values is a list of arrays (one for each input)
        # index 0: time series inputs shap values
        # index 1: tabular inputs shap values
        # shap_values[1] is what we want.
        
        # Plot
        shap.summary_plot(shap_values[1], X_tabular[100:110], feature_names=['Age', 'Parity', 'Gestation'], show=False)
        plt.savefig(os.path.join(FIGURE_DIR, "shap_summary.png"))
        print("Saved shap_summary.png")
        
    except Exception as e:
        print(f"SHAP failed: {e}")

if __name__ == "__main__":
    main()
