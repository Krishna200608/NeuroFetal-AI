import streamlit as st
import numpy as np
import tensorflow as tf
import wfdb
import os
import sys

# Add parent directory to path to allow importing 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from utils
from utils.helpers import inject_custom_css, parse_header_metadata
from utils.components import (
    render_header, 
    render_kpi_cards, 
    render_diagnosis, 
    render_signal_chart, 
    render_xai
)

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="NeuroFetal AI | Clinical Monitor",
    page_icon="assets/logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Robust path handling: Try local relative first, then Colab absolute
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Models are now in ../models relative to scripts/app.py
    local_path = os.path.join(current_dir, "..", "models", "best_model_fold_5.keras")
    colab_path = '/content/drive/MyDrive/Research_Project/Code/models/best_model_fold_5.keras'
    
    path_to_use = local_path if os.path.exists(local_path) else colab_path
    
    try:
        model = tf.keras.models.load_model(path_to_use)
        return model, path_to_use
    except Exception as e:
        st.error(f"❌ Critical Error: Model not found at {path_to_use}. Please verify paths.")
        return None, None

model, model_path_loaded = load_model()

# --- UI COMPONENTS ---

def render_sidebar():
    st.sidebar.image("assets/logo.jpg", width=80)
    st.sidebar.title("NeuroFetal AI")
    
    # Theme Toggle
    dark_mode = st.sidebar.toggle("Dark Mode", value=False)
    theme = "Dark" if dark_mode else "Light"
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Patient Data")
    
    # Initialize Session State for sliders if not exists
    if 'parity' not in st.session_state: st.session_state['parity'] = 0
    if 'gestation' not in st.session_state: st.session_state['gestation'] = 40
    if 'age' not in st.session_state: st.session_state['age'] = 30
    
    with st.sidebar.expander("Signal Data", expanded=True):
        uploaded_file = st.file_uploader("PhysioNet Record (.dat)", type=["dat"], help="Upload the raw FHR signal file")
        uploaded_header = st.file_uploader("Header File (.hea)", type=["hea"], help="Upload the clinical header file")
        
        # Auto-population Logic
        if uploaded_header:
            # Check if we already parsed this specific file to avoid overwriting user manual edits constantly
            # We use the file name as a tracker
            if 'last_loaded_header' not in st.session_state or st.session_state['last_loaded_header'] != uploaded_header.name:
                try:
                    # Use imported function
                    meta = parse_header_metadata(uploaded_header)
                    if 'age' in meta: st.session_state['age'] = meta['age']
                    if 'parity' in meta: st.session_state['parity'] = meta['parity']
                    if 'gestation' in meta: st.session_state['gestation'] = meta['gestation']
                    st.session_state['last_loaded_header'] = uploaded_header.name
                    st.toast("✅ Clinical features auto-filled from Header file!", icon="📋")
                except Exception as e:
                    st.error(f"Error parsing header: {e}")
        
        if uploaded_file:
            st.caption(f"Loaded: {uploaded_file.name}")
    
    with st.sidebar.expander("Clinical Features", expanded=True):
        # We use strict session_state keys to enable the update
        parity = st.slider("Parity", 0, 10, key="parity", help="Number of previous births")
        gestation = st.slider("Gestation (Weeks)", 20, 45, key="gestation", help="Weeks of pregnancy")
        age = st.slider("Maternal Age", 15, 50, key="age")

    # Run button
    run_btn = st.sidebar.button("Run Diagnostics", type="primary", width="stretch")  # Updated param
    
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin-top: 20px; font-size: 0.8rem; color: #888;">
        NeuroFetal AI v2.0 <br>
        Medical Device Software
    </div>
    """, unsafe_allow_html=True)
    
    return uploaded_file, uploaded_header, parity, gestation, age, run_btn, theme

# --- MAIN APP LOGIC ---

def main():
    # Render Sidebar first to get Theme
    uploaded_file, uploaded_header, parity, gestation, age, run_btn, theme = render_sidebar()
    
    # Inject CSS based on theme
    inject_custom_css(theme)
    
    # Header
    render_header()
    
    if run_btn:
        if uploaded_file and uploaded_header:
            if model is None:
                st.error("Model not loaded.")
                return

            with st.spinner("🔄 Processing signal and generating diagnosis..."):
                try:
                    # 1. Save temp files using absolute paths to avoid ambiguity
                    # We use a unique name to prevent collisions if using multiple sessions (though simple for now)
                    # Force the file extension to be consistent
                    base_name = uploaded_file.name.split('.')[0]
                    # Create absolute paths
                    temp_dir = os.path.dirname(os.path.abspath(__file__))
                    path_dat = os.path.join(temp_dir, f"{base_name}.dat")
                    path_hea = os.path.join(temp_dir, f"{base_name}.hea")
                    
                    # Write files
                    with open(path_dat, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    with open(path_hea, "wb") as f:
                        f.write(uploaded_header.getbuffer())
                    
                    # Verify existence
                    if not os.path.exists(path_dat):
                        raise FileNotFoundError(f"Failed to create {path_dat}")

                    # 2. Read Record using WFDB (pass the full path without extension)
                    record_path = os.path.join(temp_dir, base_name)
                    record = wfdb.rdrecord(record_path)
                    signal = record.p_signal[:, 0]
                    
                    # 3. Preprocessing (Strictly matching training)
                    # Training uses 20-min windows (1200 samples @ 1Hz)
                    # We will slice the input into 20-min windows with overlap, predict on all, 
                    # and pick the "Risk Window" (Max Probability) for prognosis and XAI.
                    
                    WINDOW_SIZE = 1200 # 20 mins * 60 sec
                    STRIDE = 600       # 10 mins overlap
                    
                    # Pad if shorter than one window
                    if len(signal) < WINDOW_SIZE:
                        pad = np.zeros(WINDOW_SIZE - len(signal))
                        signal = np.concatenate([pad, signal])
                        
                    # MinMax Normalize Full Signal first (for consistent visualization)
                    _min, _max = np.min(signal), np.max(signal)
                    signal_norm = (signal - _min) / (_max - _min + 1e-8)
                    
                    # Generate Sliding Windows
                    windows = []
                    window_indices = [] # Store (start, end)
                    
                    # Loop
                    num_windows = (len(signal_norm) - WINDOW_SIZE) // STRIDE + 1
                    if num_windows < 1: num_windows = 1 # Should handle via padding above, but safety
                    
                    for i in range(int(num_windows)):
                        start = i * STRIDE
                        end = start + WINDOW_SIZE
                        if end > len(signal_norm): break
                        
                        w = signal_norm[start:end]
                        windows.append(w)
                        window_indices.append((start, end))
                        
                    # Batch Prediction
                    # Prepare Tabular (duplicated for each window)
                    X_signal_batch = np.array(windows).reshape(-1, WINDOW_SIZE, 1)
                    X_tabular_batch = np.tile([parity, gestation, age], (len(windows), 1))
                    
                    predictions = model.predict([X_signal_batch, X_tabular_batch])
                    
                    # 4. Select Max Risk Window
                    max_idx = np.argmax(predictions)
                    prob = float(predictions[max_idx][0])
                    
                    # Get the specific window data for XAI
                    risk_window_signal = X_signal_batch[max_idx:max_idx+1] # Shape (1, 1200, 1)
                    risk_window_start = window_indices[max_idx][0]
                    risk_window_end = window_indices[max_idx][1]
                    
                    # 5. Render Dashboard
                    
                    # Row 1: KPI formatting
                    render_kpi_cards(prob, len(signal)/60) 
                    
                    st.divider()
                    
                    # Row 2: Diagnosis & Signal
                    c_diag, c_chart = st.columns([1, 2])
                    
                    with c_diag:
                        render_diagnosis(prob)
                        st.markdown("### Clinical Summary")
                        st.dataframe({
                            "Metric": ["Patient Age", "Gestation", "Parity", "Risk Window"],
                            "Value": [f"{age}", f"{gestation} wk", f"{parity}", f"{risk_window_start//60}-{risk_window_end//60} min"]
                        }, hide_index=True, width="stretch") 

                    with c_chart:
                        # We pass the full signal for context, but highlighting the risk window
                        # Note: We need to update render_signal_chart to accept highlight args, or just render normally.
                        # For now, simpler to render normal, but title explains.
                        render_signal_chart(signal_norm, fs=1, theme=theme)
                        st.caption(f"Analyzing {len(windows)} windows. Showing max risk segment ({prob:.1%}) found at {risk_window_start//60}m - {risk_window_end//60}m.")
                    
                    st.divider()
                    
                    # Row 3: XAI
                    # Pass ONLY the risk window (1, 1200, 1) + Tabular (1, 3) because Model expects that shape
                    render_xai(model, risk_window_signal, X_tabular_batch[max_idx:max_idx+1], {'Age': age, 'Gestation': gestation, 'Parity': parity}, theme=theme)
                    
                    # Cleanup
                    try:
                       # os.remove(path_dat) # Keep for debugging if needed, but usually good to clean
                       # os.remove(path_hea)
                       pass
                    except: pass
                    
                except Exception as e:
                    st.error(f"Analysis Failed: {str(e)}")
                    st.exception(e)
        else:
            st.warning("⚠️ Please upload both .dat and .hea files to proceed.")
            
    else:
        # Welcome / Empty State
        st.markdown("""
        <div style="text-align: center; padding: 60px; background-color: white; border-radius: 12px; border: 2px dashed #e1e4e8; margin-top: 20px;">
            <span class="material-symbols-rounded" style="font-size: 64px; color: #005eb8; margin-bottom: 20px;">monitor_heart</span>
            <h2 style="color: #2c3e50; margin-top: 10px;">Ready for Analysis</h2>
            <p style="color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                Upload patient <b>PhysioNet recordings (.dat/.hea)</b> from the sidebar to initialize the clinical support system.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()