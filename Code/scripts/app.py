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

# Import Custom Layers for Model Loading
from utils.attention_blocks import SEBlock, TemporalAttentionBlock
from utils.model import CrossModalAttention
from utils.focal_loss import FocalLoss

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
N_TABULAR_FEATURES = 16
N_CSP_FEATURES = 19

# --- REAL-TIME FEATURE EXTRACTION ---

def compute_baseline_rt(fhr, fs=1):
    """Estimate FHR baseline using a wide moving median (~10 min windows)."""
    window = int(10 * 60 * fs)
    if window < 1:
        window = 1
    half_w = window // 2
    n = len(fhr)
    baseline = np.zeros(n)
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        segment = fhr[start:end]
        valid = segment[segment > 0]
        baseline[i] = np.median(valid) if len(valid) > 0 else 0
    return baseline


def extract_realtime_tabular(fhr_window_raw, uc_window_raw, age, parity, gestation):
    """
    Extract 16 tabular features from a single FHR/UC window at inference time.
    Matches the feature vector from data_ingestion.py exactly:
      [signal_loss_pct, fhr_baseline, fhr_stv, fhr_ltv, fhr_accel_count,
       fhr_decel_count, fhr_decel_area, fhr_range, fhr_iqr, fhr_entropy,
       uc_freq, uc_intensity_mean, fhr_uc_lag, age, parity, gestation]
    """
    from scipy.signal import find_peaks

    valid = fhr_window_raw[fhr_window_raw > 0]
    
    # Signal loss
    signal_loss = np.mean(fhr_window_raw == 0)
    
    # Baseline
    baseline = compute_baseline_rt(fhr_window_raw)
    valid_bl = baseline[baseline > 0]
    fhr_baseline = float(np.mean(valid_bl)) if len(valid_bl) > 0 else 0.0
    
    # STV: mean absolute first-order difference
    fhr_stv = float(np.mean(np.abs(np.diff(valid)))) if len(valid) > 1 else 0.0
    
    # LTV: std of 1-minute segment means
    seg_len = 60
    n_segs = len(fhr_window_raw) // seg_len
    if n_segs >= 2:
        means = []
        for i in range(n_segs):
            seg = fhr_window_raw[i * seg_len:(i + 1) * seg_len]
            v = seg[seg > 0]
            if len(v) > 0:
                means.append(np.mean(v))
        fhr_ltv = float(np.std(means)) if len(means) >= 2 else 0.0
    else:
        fhr_ltv = 0.0
    
    # Accelerations: rises > 15 bpm above baseline for > 15s
    diff_above = fhr_window_raw - baseline
    above = diff_above > 15
    runs = np.diff(np.concatenate(([0], above.astype(int), [0])))
    starts_a = np.where(runs == 1)[0]
    ends_a = np.where(runs == -1)[0]
    accel_count = sum(1 for s, e in zip(starts_a, ends_a) if (e - s) >= 15)
    
    # Decelerations: dips > 15 bpm below baseline for > 15s
    diff_below = baseline - fhr_window_raw
    valid_mask = fhr_window_raw > 0
    below = (diff_below > 15) & valid_mask
    runs_d = np.diff(np.concatenate(([0], below.astype(int), [0])))
    starts_d = np.where(runs_d == 1)[0]
    ends_d = np.where(runs_d == -1)[0]
    decel_count = 0
    decel_area = 0.0
    for s, e in zip(starts_d, ends_d):
        if (e - s) >= 15:
            decel_count += 1
            decel_area += float(np.sum(diff_below[s:e]))
    
    # Range & IQR
    fhr_range = float(np.max(valid) - np.min(valid)) if len(valid) > 0 else 0.0
    fhr_iqr = float(np.percentile(valid, 75) - np.percentile(valid, 25)) if len(valid) > 0 else 0.0
    
    # Entropy (simplified for speed at inference)
    fhr_entropy = 0.0
    if len(valid) > 50:
        # Approximate entropy via normalized std
        fhr_entropy = float(np.log(np.std(valid) + 1e-8))
    
    # UC features
    uc_freq = 0.0
    uc_intensity = 0.0
    if uc_window_raw is not None and len(uc_window_raw) > 10:
        uc_smooth = np.convolve(uc_window_raw, np.ones(30) / 30, mode='same')
        threshold = np.mean(uc_smooth) + 0.3 * np.std(uc_smooth)
        peaks, _ = find_peaks(uc_smooth, height=threshold, distance=120, prominence=0.1)
        uc_freq = float(len(peaks))
        uc_intensity = float(np.mean(uc_smooth[peaks])) if len(peaks) > 0 else 0.0
    
    # FHR-UC lag
    fhr_uc_lag = 0.0
    if uc_window_raw is not None and np.std(fhr_window_raw) > 0 and np.std(uc_window_raw) > 0:
        fhr_n = (fhr_window_raw - np.mean(fhr_window_raw)) / (np.std(fhr_window_raw) + 1e-8)
        uc_n = (uc_window_raw - np.mean(uc_window_raw)) / (np.std(uc_window_raw) + 1e-8)
        max_lag = 300  # 5 minutes
        corr = np.correlate(fhr_n, uc_n, mode='full')
        mid = len(corr) // 2
        start = max(0, mid - max_lag)
        end = min(len(corr), mid + max_lag + 1)
        corr_window = corr[start:end]
        if len(corr_window) > 0:
            lag_idx = np.argmax(np.abs(corr_window)) - (end - start) // 2
            fhr_uc_lag = float(lag_idx)
    
    # Assemble 16-feature vector (matches training order)
    features = np.array([
        signal_loss, fhr_baseline, fhr_stv, fhr_ltv,
        accel_count, decel_count, decel_area,
        fhr_range, fhr_iqr, fhr_entropy,
        uc_freq, uc_intensity, fhr_uc_lag,
        float(age), float(parity), float(gestation)
    ], dtype=np.float32)
    
    return features


def extract_realtime_csp(fhr_window, uc_window):
    """
    Extract 19 CSP-based features from FHR + UC signals.
    Uses statistical features when CSP filters are not fitted (inference mode).
    Feature vector: [fhr_mean, fhr_std, fhr_min, fhr_max, fhr_mad, fhr_beta0,
                     fhr_skewness, fhr_kurtosis, fhr_sqi, uc_mean, uc_std, uc_count,
                     cross_corr_max, cross_corr_mean, labor_stage_flag, csp_0..csp_3]
    """
    from scipy.stats import skew, kurtosis
    from scipy.signal import correlate
    
    features = []
    
    # FHR basic stats
    features.append(np.mean(fhr_window))
    features.append(np.std(fhr_window))
    features.append(np.min(fhr_window))
    features.append(np.max(fhr_window))
    
    # FHR advanced: MAD
    features.append(np.median(np.abs(fhr_window - np.median(fhr_window))))
    
    # Beta0 (baseline intercept)
    n = len(fhr_window)
    if n > 1:
        x = np.arange(n)
        slope = (n * np.sum(x * fhr_window) - np.sum(x) * np.sum(fhr_window)) / \
                (n * np.sum(x ** 2) - np.sum(x) ** 2 + 1e-8)
        beta0 = np.mean(fhr_window) - slope * np.mean(x)
        features.append(beta0)
    else:
        features.append(0.0)
    
    # Skewness & Kurtosis
    features.append(skew(fhr_window) if len(fhr_window) > 2 else 0.0)
    features.append(kurtosis(fhr_window) if len(fhr_window) > 2 else 0.0)
    
    # SQI (signal quality: fraction in valid range)
    valid_frac = np.mean((fhr_window > 0.05) & (fhr_window < 0.95))  # normalized range
    features.append(valid_frac)
    
    # UC features
    if uc_window is not None and len(uc_window) > 0:
        features.append(np.mean(uc_window))
        features.append(np.std(uc_window))
        features.append(float(np.sum(uc_window > 0.5)))
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Cross-correlation
    if uc_window is not None and len(uc_window) > 0:
        cross_corr = correlate(fhr_window, uc_window, mode='same')
        features.append(np.max(np.abs(cross_corr)))
        features.append(np.mean(np.abs(cross_corr)))
    else:
        features.extend([0.0, 0.0])
    
    # Labor stage flag (assume last window = potential stage 2)
    features.append(0.0)
    
    # CSP log-variance features (4 components — zeros when not fitted)
    features.extend([0.0, 0.0, 0.0, 0.0])
    
    return np.array(features[:N_CSP_FEATURES], dtype=np.float32)


# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Load the enhanced 3-input model with fallback to legacy model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "..", "models")
    
    # Enhanced model paths (3-input: FHR + tabular_16 + CSP_19)
    enhanced_local = os.path.join(model_dir, "enhanced_model_fold_1.keras")
    enhanced_colab = '/content/drive/MyDrive/Research_Project/Code/models/enhanced_model_fold_1.keras'
    
    # Legacy model paths (2-input: FHR + tabular_3)
    legacy_local = os.path.join(model_dir, "best_model_fold_5.keras")
    legacy_colab = '/content/drive/MyDrive/Research_Project/Code/models/best_model_fold_5.keras'
    
    custom_objects = {
        'SEBlock': SEBlock,
        'TemporalAttentionBlock': TemporalAttentionBlock,
        'CrossModalAttention': CrossModalAttention,
        'FocalLoss': FocalLoss,
        'focal_loss_fixed': FocalLoss(gamma=2.5, alpha=0.75)
    }
    
    model = None
    path_used = None
    is_enhanced = False
    
    # Try enhanced model first
    for path in [enhanced_local, enhanced_colab]:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                path_used = path
                is_enhanced = True
                break
            except Exception as e:
                print(f"Warning: Enhanced model load failed ({path}): {e}")
    
    # Fallback to legacy model
    if model is None:
        for path in [legacy_local, legacy_colab]:
            if os.path.exists(path):
                try:
                    model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                    path_used = path
                    is_enhanced = False
                    break
                except Exception as e:
                    print(f"Warning: Legacy model load failed ({path}): {e}")
    
    if model is None:
        st.error("Critical Error: No model found. Please verify model files exist in Code/models/")
    
    return model, path_used, is_enhanced


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

            with st.spinner("Processing signal and extracting features..."):
                try:
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
                        
                        # Extract UC channel if available (for CSP + tabular features)
                        if record.p_signal.shape[1] > 1:
                            uc_signal = record.p_signal[:, 1].copy()
                        
                        # Store raw signal for feature extraction (before normalization)
                        signal_raw = signal.copy()
                        uc_raw = uc_signal.copy() if uc_signal is not None else None
                    
                    # --- PREPROCESSING ---
                    # Pad if shorter than one window
                    if len(signal) < WINDOW_SIZE:
                        pad = np.zeros(WINDOW_SIZE - len(signal))
                        signal = np.concatenate([pad, signal])
                        signal_raw = np.concatenate([pad, signal_raw])
                        if uc_raw is not None:
                            uc_raw = np.concatenate([np.zeros(WINDOW_SIZE - len(uc_raw)), uc_raw])
                    
                    # Normalize FHR for model input (exclude zeros)
                    valid_mask = signal > 0
                    if np.sum(valid_mask) > 0:
                        _min = np.min(signal[valid_mask])
                        _max = np.max(signal[valid_mask])
                        signal_norm = np.zeros_like(signal)
                        signal_norm[valid_mask] = (signal[valid_mask] - _min) / (_max - _min + 1e-8)
                    else:
                        signal_norm = np.zeros_like(signal)
                    
                    # Normalize UC if available
                    uc_norm = None
                    if uc_raw is not None:
                        uc_valid = uc_raw > 0
                        if np.sum(uc_valid) > 0:
                            uc_min = np.min(uc_raw[uc_valid])
                            uc_max = np.max(uc_raw[uc_valid])
                            uc_norm = np.zeros_like(uc_raw)
                            uc_norm[uc_valid] = (uc_raw[uc_valid] - uc_min) / (uc_max - uc_min + 1e-8)
                        else:
                            uc_norm = np.zeros_like(uc_raw)
                    
                    # --- GENERATE SLIDING WINDOWS ---
                    windows_fhr = []
                    windows_tabular = []
                    windows_csp = []
                    window_indices = []
                    
                    num_windows = max(1, (len(signal_norm) - WINDOW_SIZE) // STRIDE + 1)
                    
                    for i in range(int(num_windows)):
                        start = i * STRIDE
                        end = start + WINDOW_SIZE
                        if end > len(signal_norm):
                            break
                        
                        # FHR window (normalized for model)
                        w_fhr = signal_norm[start:end]
                        windows_fhr.append(w_fhr)
                        window_indices.append((start, end))
                        
                        if is_enhanced_model:
                            # Extract raw-signal features for tabular input
                            w_raw = signal_raw[start:end] if start < len(signal_raw) else np.zeros(WINDOW_SIZE)
                            w_uc_raw = uc_raw[start:end] if uc_raw is not None and start < len(uc_raw) else None
                            
                            tab_feat = extract_realtime_tabular(w_raw, w_uc_raw, age, parity, gestation)
                            windows_tabular.append(tab_feat)
                            
                            # CSP features from normalized signals
                            w_uc_norm = uc_norm[start:end] if uc_norm is not None and start < len(uc_norm) else None
                            csp_feat = extract_realtime_csp(w_fhr, w_uc_norm)
                            windows_csp.append(csp_feat)
                        else:
                            # Legacy model: 3 tabular features
                            windows_tabular.append(np.array([parity, gestation, age], dtype=np.float32))
                    
                    # --- BATCH PREDICTION ---
                    X_signal_batch = np.array(windows_fhr).reshape(-1, WINDOW_SIZE, 1)
                    X_tabular_batch = np.array(windows_tabular)
                    
                    if is_enhanced_model:
                        X_csp_batch = np.array(windows_csp)
                        # Handle NaNs
                        X_tabular_batch = np.nan_to_num(X_tabular_batch, nan=0.0)
                        X_csp_batch = np.nan_to_num(X_csp_batch, nan=0.0)
                        
                        predictions = model.predict(
                            [X_signal_batch, X_tabular_batch, X_csp_batch], verbose=0
                        )
                    else:
                        predictions = model.predict(
                            [X_signal_batch, X_tabular_batch], verbose=0
                        )
                    
                    # Handle multi-output models (main output + auxiliary pH head)
                    if isinstance(predictions, list):
                        predictions = predictions[0]
                    
                    # --- SELECT MAX RISK WINDOW ---
                    max_idx = np.argmax(predictions)
                    prob = float(predictions[max_idx][0])
                    
                    risk_window_signal = X_signal_batch[max_idx:max_idx+1]
                    risk_window_start = window_indices[max_idx][0]
                    risk_window_end = window_indices[max_idx][1]
                    
                    # --- RENDER DASHBOARD ---
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