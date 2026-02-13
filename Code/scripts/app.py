import streamlit as st
import numpy as np
import tensorflow as tf
import wfdb
import os
import sys
import pickle

# Add parent directory to path to allow importing 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from utils
from utils.helpers import inject_custom_css, parse_header_metadata
from utils.components import (
    render_header, 
    render_kpi_cards, 
    render_diagnosis, 
    render_signal_chart, 
    render_xai,
    render_mismatch_error,
    render_uncertainty_analysis
)

# Import Modularized Components
from utils.model_loader import load_model
from utils.feature_extractor import extract_18_tabular_rt, extract_realtime_csp




# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="NeuroFetal AI | Clinical Monitor",
    page_icon="assets/Logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
WINDOW_SIZE = 1200  # 20 mins * 60 sec @ 1Hz
STRIDE = 600        # 10 mins overlap
N_TABULAR_FEATURES = 18 # Changed from 16 to match SOTA model
N_CSP_FEATURES = 19

# --- REAL-TIME FEATURE EXTRACTION ---

# Feature extraction functions moved to utils/feature_extractor.py


# --- MODEL LOADING ---
# Model loading moved to utils/model_loader.py


model, model_path_loaded, is_enhanced_model = load_model()

# --- UI COMPONENTS ---

def render_sidebar():
    # Robust Path for Logo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, "..", "assets", "Logo-black.png")
    
    st.sidebar.image(logo_path, width=120)
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
            if 'last_loaded_header' not in st.session_state or st.session_state['last_loaded_header'] != uploaded_header.name:
                try:
                    meta = parse_header_metadata(uploaded_header)
                    if 'age' in meta: st.session_state['age'] = meta['age']
                    if 'parity' in meta: st.session_state['parity'] = meta['parity']
                    if 'gestation' in meta: st.session_state['gestation'] = meta['gestation']
                    st.session_state['last_loaded_header'] = uploaded_header.name
                    st.toast("Clinical features auto-filled from Header file!")
                except Exception as e:
                    st.error(f"Error parsing header: {e}")
        
        if uploaded_file:
            st.caption(f"Loaded: {uploaded_file.name}")
    
    with st.sidebar.expander("Clinical Features", expanded=True):
        parity = st.slider("Parity", 0, 10, key="parity", help="Number of previous births")
        gestation = st.slider("Gestation (Weeks)", 20, 45, key="gestation", help="Weeks of pregnancy")
        age = st.slider("Maternal Age", 15, 50, key="age")

    # Run button
    run_btn = st.sidebar.button("Run Diagnostics", type="primary", use_container_width=True)
    
    # Model info badge
    model_type = "Enhanced 3-Input Ensemble" if is_enhanced_model else "Legacy 2-Input"
    
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin-top: 20px; font-size: 0.8rem; color: #888;">
        NeuroFetal AI v4.0 <br>
        {model_type} <br>
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
    render_header(theme)
    
    if run_btn:
        if uploaded_file and uploaded_header:
            
            # --- VALIDATION: Check for filename mismatch ---
            base_name_dat = os.path.splitext(uploaded_file.name)[0]
            base_name_hea = os.path.splitext(uploaded_header.name)[0]
            
            if base_name_dat != base_name_hea:
                render_mismatch_error(uploaded_file.name, uploaded_header.name, theme)
                return
            # ---------------------------------------------

            if model is None:
                st.error("Model not loaded.")
                return

            try:
                # --- PROCESSING PHASE (Inside Status) ---
                with st.status("🚀 Initiating Neural Analysis Sequence...", expanded=True) as status:
                    
                    # 1. File Processing
                    st.write("🔹 Parsing PhysioNet data files...")
                    import tempfile
                    
                    signal = None
                    uc_signal = None
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        base_name = base_name_dat
                        path_dat = os.path.join(temp_dir, f"{base_name}.dat")
                        path_hea = os.path.join(temp_dir, f"{base_name}.hea")
                        
                        with open(path_dat, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        with open(path_hea, "wb") as f:
                            f.write(uploaded_header.getbuffer())
                        
                        record_path = os.path.join(temp_dir, base_name)
                        record = wfdb.rdrecord(record_path)
                        signal = record.p_signal[:, 0].copy()  # FHR channel
                        if record.p_signal.shape[1] > 1:
                            uc_signal = record.p_signal[:, 1].copy()
                        
                        signal_raw = signal.copy()
                        uc_raw = uc_signal.copy() if uc_signal is not None else None

                    # 2. Windowing
                    st.write("🔹 Segmenting signal into 20-minute clinical windows...")
                    if len(signal) < WINDOW_SIZE:
                        pad = np.zeros(WINDOW_SIZE - len(signal))
                        signal = np.concatenate([pad, signal])
                        signal_raw = np.concatenate([pad, signal_raw])
                        if uc_raw is not None:
                            uc_raw = np.concatenate([np.zeros(WINDOW_SIZE - len(uc_raw)), uc_raw])
                    
                    windows_fhr = []
                    windows_tabular = []
                    windows_csp = []
                    window_indices = []
                    timestamps = []
                    
                    num_windows = max(1, (len(signal) - WINDOW_SIZE) // STRIDE + 1)
                    
                    for i in range(int(num_windows)):
                        start = i * STRIDE
                        end = start + WINDOW_SIZE
                        if end > len(signal): break
                        
                        fhr_window_raw = signal_raw[start:end]
                        uc_window_raw = uc_raw[start:end] if uc_raw is not None else np.zeros(WINDOW_SIZE)
                        
                        if len(fhr_window_raw) == WINDOW_SIZE:
                            fhr_window_norm = normalize_fhr(fhr_window_raw)
                            csp_features = extract_realtime_csp(fhr_window_raw, uc_window_raw)
                            current_header = {'Age': age, 'Parity': parity, 'Gestation': gestation, 'Gravidity': 1, 'Weight': 70}
                            tab_features = extract_18_tabular_rt(fhr_window_raw, uc_window_raw, current_header)
                            
                            windows_fhr.append(fhr_window_norm)
                            windows_tabular.append(tab_features)
                            windows_csp.append(csp_features)
                            timestamps.append(i * (STRIDE / 3600.0))
                            window_indices.append((start, end))
                    
                    st.write(f"🔹 Extracted {len(windows_fhr)} windows. Computing {N_CSP_FEATURES}x Spatial Filters per window...")

                    # 3. Batching & Prediction
                    X_signal_batch = np.array(windows_fhr).reshape(-1, WINDOW_SIZE, 1)
                    X_tabular_batch = np.array(windows_tabular)
                    
                    if is_enhanced_model:
                        st.write("🔹 Running Enhanced Ensemble Inference (ResNet + Inception + XGBoost)...")
                        X_csp_batch = np.array(windows_csp)
                        X_tabular_batch = np.nan_to_num(X_tabular_batch, nan=0.0)
                        X_csp_batch = np.nan_to_num(X_csp_batch, nan=0.0)
                        
                        # Normalization logic
                        try:
                            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            processed_dir = os.path.join(base_dir, "..", "Datasets", "processed")
                            tab_means = np.load(os.path.join(processed_dir, "tabular_means.npy"))
                            tab_stds = np.load(os.path.join(processed_dir, "tabular_stds.npy"))
                            X_tabular_batch = (X_tabular_batch - tab_means) / (tab_stds + 1e-8)
                        except Exception:
                            st.warning("⚠️ Using un-normalized features (Stats missing).")
                        
                        predictions = model.predict([X_signal_batch, X_tabular_batch, X_csp_batch], verbose=0)
                    else:
                        st.write("🔹 Running Legacy Inference Model...")
                        predictions = model.predict([X_signal_batch, X_tabular_batch], verbose=0)
                    
                    if isinstance(predictions, list): predictions = predictions[0]

                    st.write("✅ Diagnostics Complete. Finalizing visualization...")
                    status.update(label="Analysis Complete", state="complete", expanded=False)

                    
                # --- SELECT MAX RISK WINDOW ---
                max_idx = np.argmax(predictions)
                prob = float(predictions[max_idx][0])
                
                risk_window_signal = X_signal_batch[max_idx:max_idx+1]
                risk_window_start = window_indices[max_idx][0]
                risk_window_end = window_indices[max_idx][1]
                
                
                # --- RENDER DASHBOARD ---
                # Create normalized signal for visualization (0-1 range)
                signal_norm = normalize_fhr(signal)

                render_kpi_cards(prob, len(signal)/60)
                
                st.divider()
                
                c_diag, c_chart = st.columns([1, 2])
                
                with c_diag:
                    render_diagnosis(prob, theme)
                    st.markdown("### Clinical Summary")
                    
                    summary_data = {
                        "Metric": ["Patient Age", "Gestation", "Parity", "Risk Window", "Model"],
                        "Value": [
                            f"{age}",
                            f"{gestation} wk",
                            f"{parity}",
                            f"{risk_window_start//60}-{risk_window_end//60} min",
                            "Enhanced Ensemble" if is_enhanced_model else "Legacy"
                        ]
                    }
                    st.dataframe(summary_data, hide_index=True, use_container_width=True)

                with c_chart:
                    render_signal_chart(signal_norm, fs=1, theme=theme)
                    st.caption(
                        f"Analyzing {len(windows_fhr)} windows. "
                        f"Showing max risk segment ({prob:.1%}) "
                        f"found at {risk_window_start//60}m - {risk_window_end//60}m."
                    )
                
                st.divider()
                
                # --- XAI ---
                if is_enhanced_model:
                    risk_tabular = X_tabular_batch[max_idx:max_idx+1]
                    risk_csp = X_csp_batch[max_idx:max_idx+1]
                    render_xai(
                        model, risk_window_signal, risk_tabular, 
                        {'Age': age, 'Gestation': gestation, 'Parity': parity},
                        theme=theme, csp_input=risk_csp
                    )
                else:
                    render_xai(
                        model, risk_window_signal, X_tabular_batch[max_idx:max_idx+1],
                        {'Age': age, 'Gestation': gestation, 'Parity': parity},
                        theme=theme
                    )
                
            except Exception as e:
                st.error(f"Analysis Failed: {str(e)}")
                st.exception(e)
        else:
            st.warning("Please upload both .dat and .hea files to proceed.")
            
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

    # --- FOOTER / MODEL METRICS ---
    st.divider()
    
    with st.expander("Model Validation & Uncertainty Metrics (Static Report)", icon=":material/analytics:"):
        st.caption("Model Reliability Analysis")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        report_dir = os.path.join(current_dir, "..", "..", "Reports", "uncertainty_analysis", "fold_1")
        render_uncertainty_analysis(report_dir)

if __name__ == "__main__":
    main()