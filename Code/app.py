import streamlit as st
import numpy as np
import tensorflow as tf
import wfdb
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION & STYLING ---
# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="NeuroFetal AI | Clinical Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css(theme="Light"):
    """
    Injects CSS based on the selected theme.
    """
    if theme == "Dark":
        # Dark Mode CSS
        background_color = "#0e1117"
        text_color = "#fafafa"
        card_bg = "#262730"
        card_border = "#3d3e45"
        text_secondary = "#a0a5ab"
        header_bg = "#262730"
        shadow_opacity = "0.3"
    else:
        # Light Mode CSS (Default)
        background_color = "#f4f6f9"
        text_color = "#333333"
        card_bg = "#ffffff"
        card_border = "#eaecf0"
        text_secondary = "#64748b"
        header_bg = "#ffffff"
        shadow_opacity = "0.05"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');

        /* Global Font & Colors */
        html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame {{
            font-family: 'Inter', sans-serif;
            color: {text_color} !important;
        }}
        
        /* Specific fix for Metrics in Light Mode */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
            color: {text_color} !important;
        }}
        
        /* Fix for Tab visibility */
        button[data-baseweb="tab"] {{
            color: {text_secondary} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {text_color} !important;
            font-weight: 600 !important;
        }}
        
        /* Icon alignment fix logic */
        .material-symbols-rounded {{
            vertical-align: middle;
            font-size: 1.2rem;
            position: relative;
            top: -1px;
        }}

        /* Main Background */
        .stApp {{
            background-color: {background_color};
            color: {text_color};
        }}

        /* Header Container */
        .header-container {{
            background-color: {header_bg};
            padding: 1.5rem 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,{shadow_opacity});
            margin-bottom: 25px;
            border-left: 6px solid #005eb8; /* NHS Blue - Constant */
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .app-title {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {'#e1f5fe' if theme=='Dark' else '#003366'} !important;
            margin: 0;
            letter-spacing: -0.5px;
        }}
        
        .app-subtitle {{
            font-size: 1rem;
            color: {text_secondary} !important;
            font-weight: 400;
            margin-top: 5px;
        }}

        /* Status Badges */
        .status-pill {{
            padding: 6px 16px;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-left: 10px;
        }}
        .status-online {{ background-color: #e6f4ea; color: #137333 !important; border: 1px solid #ceead6; }}
        .status-info {{ background-color: #e8f0fe; color: #1967d2 !important; border: 1px solid #d2e3fc; }}

        /* Custom Metric Card */
        .metric-card {{
            background-color: {card_bg};
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,{shadow_opacity});
            border: 1px solid {card_border};
            text-align: center;
            transition: transform 0.2s;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,{int(float(shadow_opacity)*2)});
        }}
        .metric-label {{
            color: {text_secondary} !important;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .metric-value {{
            color: {text_color} !important;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
        }}
        .metric-delta {{
            font-size: 0.85rem;
            margin-top: 5px;
            color: {text_secondary}; 
        }}
        .text-green {{ color: #16a34a !important; }}
        .text-red {{ color: #dc2626 !important; }}

        /* Diagnosis Card */
        .diagnosis-container {{
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,{shadow_opacity});
        }}
        .diag-safe {{
            background: { 'linear-gradient(135deg, #1e1e1e 0%, #064e3b 100%)' if theme=='Dark' else 'linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%)' };
            border: 2px solid #bbf7d0;
        }}
        .diag-danger {{
            background: { 'linear-gradient(135deg, #1e1e1e 0%, #7f1d1d 100%)' if theme=='Dark' else 'linear-gradient(135deg, #ffffff 0%, #fef2f2 100%)' };
            border: 2px solid #fecaca;
        }}
        
        /* Remove streamlit branding stuff */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Robust path handling: Try local relative first, then Colab absolute
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(current_dir, "models", "best_model_fold_5.h5")
    colab_path = '/content/drive/MyDrive/Research_Project/Code/models/best_model_fold_5.h5'
    
    path_to_use = local_path if os.path.exists(local_path) else colab_path
    
    try:
        model = tf.keras.models.load_model(path_to_use)
        return model, path_to_use
    except Exception as e:
        st.error(f"❌ Critical Error: Model not found at {path_to_use}. Please verify paths.")
        return None, None

model, model_path_loaded = load_model()

# --- UI COMPONENTS ---


def render_header():
    st.markdown("""
    <div class="header-container">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="background: #e1f5fe; padding: 10px; border-radius: 12px; color: #0288d1;">
                <span class="material-symbols-rounded" style="font-size: 32px;">cardiology</span>
            </div>
            <div>
                <h1 class="app-title">NeuroFetal AI</h1>
                <p class="app-subtitle">Clinical Decision Support System v2.0</p>
            </div>
        </div>
        <div style="display: flex; gap: 10px;">
            <div class="status-pill status-online">
                <span class="material-symbols-rounded" style="font-size: 16px;">check_circle</span> System Online
            </div>
            <div class="status-pill status-info">
                <span class="material-symbols-rounded" style="font-size: 16px;">neurology</span> Fusion ResNet
            </div>
            <div class="status-pill status-info">
                <span class="material-symbols-rounded" style="font-size: 16px;">bolt</span> Edge Optimized
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=60)
    st.sidebar.title("NeuroFetal AI")
    
    # Theme Toggle
    # User requested a toggle switch for Dark Mode
    dark_mode = st.sidebar.toggle("Dark Mode", value=False)
    theme = "Dark" if dark_mode else "Light"
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Patient Data")
    
    with st.sidebar.expander("Signal Data", expanded=True):
        uploaded_file = st.file_uploader("PhysioNet Record (.dat)", type=["dat"], help="Upload the raw FHR signal file")
        uploaded_header = st.file_uploader("Header File (.hea)", type=["hea"], help="Upload the clinical header file")
        
        if uploaded_file:
            st.caption(f"Loaded: {uploaded_file.name}")
    
    with st.sidebar.expander("Clinical Features", expanded=True):
        parity = st.slider("Parity", 0, 10, 0, help="Number of previous births")
        gestation = st.slider("Gestation (Weeks)", 20, 45, 40, help="Weeks of pregnancy")
        age = st.slider("Maternal Age", 15, 50, 30)

    # Run button
    run_btn = st.sidebar.button("Run Diagnostics", type="primary", width="stretch")  # Updated param
    
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin-top: 20px; font-size: 0.8rem; color: #888;">
        NeuroFetal AI v2.0 <br>
        Medical Device Software
    </div>
    """, unsafe_allow_html=True)
    
    return uploaded_file, uploaded_header, parity, gestation, age, run_btn, theme

def render_kpi_cards(prob, signal_dur_min):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label"><span class="material-symbols-rounded" style="font-size:18px; color:#64748b; margin-right:5px;">emergency</span> Risk Probability</div>
            <div class="metric-value">{prob:.1%}</div>
            <div class="metric-delta">Model Output</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        risk_label = "High Risk" if prob > 0.5 else "Normal"
        color_class = "text-red" if prob > 0.5 else "text-green"
        icon = "warning" if prob > 0.5 else "check_circle"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label"><span class="material-symbols-rounded" style="font-size:18px; color:#64748b; margin-right:5px;">{icon}</span> Classification</div>
            <div class="metric-value {color_class}">{risk_label}</div>
            <div class="metric-delta">Threshold: 50%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        confidence = prob if prob > 0.5 else (1 - prob)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label"><span class="material-symbols-rounded" style="font-size:18px; color:#64748b; margin-right:5px;">verified_user</span> Model Confidence</div>
            <div class="metric-value">{confidence:.1%}</div>
            <div class="metric-delta">Reliability Score</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label"><span class="material-symbols-rounded" style="font-size:18px; color:#64748b; margin-right:5px;">timer</span> Signal Duration</div>
            <div class="metric-value">{signal_dur_min:.0f} min</div>
            <div class="metric-delta">Analysis Window</div>
        </div>
        """, unsafe_allow_html=True)

def render_diagnosis(prob):
    if prob > 0.5:
        html = f"""
        <div class="diagnosis-container diag-danger">
            <h2 style="color: #991b1b; margin-top:0;">
                <span class="material-symbols-rounded" style="vertical-align:bottom; font-size: 28px;">warning</span> 
                PATHOLOGICAL TRACE
            </h2>
            <p style="color: #7f1d1d; font-size: 1.1rem;">High probability of Fetal Compromise detected.</p>
            <h1 style="font-size: 3.5rem; margin: 10px 0; color: #b91c1c;">{prob:.1%}</h1>
            <p style="color: #ef4444; font-weight: 600;">IMMEDIATE CLINICAL REVIEW REQUIRED</p>
        </div>
        """
    else:
        html = f"""
        <div class="diagnosis-container diag-safe">
            <h2 style="color: #166534; margin-top:0;">
                <span class="material-symbols-rounded" style="vertical-align:bottom; font-size: 28px;">check_circle</span> 
                PHYSIOLOGICAL TRACE
            </h2>
            <p style="color: #14532d; font-size: 1.1rem;">Fetal well-being likely reassured.</p>
            <h1 style="font-size: 3.5rem; margin: 10px 0; color: #15803d;">{prob:.1%}</h1>
            <p style="color: #22c55e; font-weight: 600;">CONTINUE ROUTINE MONITORING</p>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

def render_signal_chart(signal, fs=1, theme="Light"):
    # Use Plotly for professional interactive look
    time_sec = np.arange(len(signal)) / fs
    time_min = time_sec / 60
    
    # Theme Logic
    is_dark = theme == "Dark"
    template = "plotly_dark" if is_dark else "plotly_white"
    line_color = '#4FC3F7' if is_dark else '#0072ce' # Light Cyan vs NHS Blue
    fill_color = 'rgba(79, 195, 247, 0.1)' if is_dark else 'rgba(0, 114, 206, 0.05)'
    
    # Create interactive plot
    fig = go.Figure()
    
    # Add Signal Trace
    fig.add_trace(go.Scatter(
        x=time_min, 
        y=signal, 
        mode='lines', 
        name='FHR Signal',
        line=dict(color=line_color, width=1.5),
        fill='tozeroy',
        fillcolor=fill_color
    ))
    
    # Layout enhancements
    fig.update_layout(
        title=dict(text="<b>Fetal Heart Rate (FHR) Analysis</b>", font=dict(size=18)),
        xaxis_title="Time (Minutes)",
        yaxis_title="Normalized Amplitude",
        template=template,
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified"
    )
    
    if not is_dark:
        # Add Grid for light mode explicitly if needed, but template usually handles it
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eee')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee')
    else:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')
    
    st.plotly_chart(fig, width='stretch')

def render_xai(model, signal, tabular, inputs_dict, theme="Light"):
    st.subheader("Explainable AI (XAI) Analysis")
    st.markdown("This section visualizes the specific features triggering the model's decision.")
    
    tab1, tab2 = st.tabs(["Grad-CAM (Signal Attention)", "Clinical Feature Contribution"])
    
    with tab1:
        st.markdown("**Gradient-weighted Class Activation Mapping (Grad-CAM)** highlights temporal regions of the FHR signal that most strongly influenced the prediction.")
        
        try:
            # Grad-CAM Logic
            last_conv_layer = [layer for layer in model.layers if 'conv' in layer.name][-1]
            grad_model = tf.keras.models.Model(
                model.inputs, [last_conv_layer.output, model.output] # Fixed input structure
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model([signal, tabular])
                
                # Robustness for Keras versions returning lists
                if isinstance(conv_outputs, list):
                    conv_outputs = conv_outputs[0]
                if isinstance(predictions, list):
                    predictions = predictions[0]
                    
                loss = predictions[:, 0]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
            
            # Resize heatmap
            heatmap_resized = tf.image.resize(heatmap.numpy().reshape(1, -1, 1), (1, signal.shape[1]))[0, :, 0].numpy()
            
            # Plotly Heatmap Overlay
            time_min = np.arange(len(signal[0,:,0])) / 60
            
            # Theme Settings
            is_dark = theme == "Dark"
            template = "plotly_dark" if is_dark else "plotly_white"
            signal_color = '#4FC3F7' if is_dark else '#0072ce' # Cyan vs Blue
            mask_color = '#FF5252' if is_dark else '#dc2626'   # Bright Red vs Deep Red
            
            
            # Interactive Combined Plot
            fig_cam = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Signal
            fig_cam.add_trace(iso_fhr := go.Scatter(
                x=time_min, y=signal[0,:,0], mode='lines', name='FHR Signal',
                line=dict(color=signal_color, width=1.5)
            ), secondary_y=False)
            
            # Heatmap (Area)
            # Thresholding for visibility using red highlight
            high_attention_mask = heatmap_resized > 0.4
            
            # Add highlights
            fig_cam.add_trace(go.Scatter(
                x=time_min, 
                y=np.where(high_attention_mask, signal[0,:,0], None),
                mode='lines',
                name='High Attention Zone',
                line=dict(color=mask_color, width=2.5),
                connectgaps=False
            ), secondary_y=False)

            fig_cam.add_trace(go.Scatter(
                x=time_min, y=heatmap_resized, mode='lines', name='Attention Score',
                line=dict(color='rgba(255,0,0,0.5)', width=0, dash='dot'),
                fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'
            ), secondary_y=True)
            
            fig_cam.update_layout(
                title="AI Focus Areas (Red regions indicate patterns triggering risk)",
                template=template,
                height=350,
                hovermode="x unified"
            )
            fig_cam.update_yaxes(title_text="FHR Amplitude", secondary_y=False)
            fig_cam.update_yaxes(title_text="Attention Intensity", secondary_y=True, showgrid=False)
            
            st.plotly_chart(fig_cam, width='stretch')
            
            
        except Exception as e:
            st.error(f"Could not generate Grad-CAM: {e}")
            
    with tab2:
        st.markdown("**Tabular Feature Analysis**: Comparison of patient vitals vs risk thresholds.")
        
        # Simple Visualization of tabular inputs relative to risk factors
        # Age > 35 is risk, Parity > 0 is usually protective or neutral depending, Gestation < 37 is risk
        
        age = inputs_dict['Age']
        gest = inputs_dict['Gestation']
        parity = inputs_dict['Parity']
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric("Maternal Age", f"{age} yrs", delta="-Risk" if age > 35 else "Normal", delta_color="inverse")
            st.progress(min(age/50, 1.0))
        with c2:
            st.metric("Gestation", f"{gest} wks", delta="-Preterm" if gest < 37 else "Term", delta_color="inverse")
            st.progress(min(gest/42, 1.0))
        with c3:
            st.metric("Parity", str(parity))
            st.progress(min(parity/5, 1.0))

        st.info("ℹ️ The Dense Tabular Branch of Fusion ResNet processes these features in parallel with the signal to adjust the final risk probability.")

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
                    target_len = 3600
                    if len(signal) > target_len:
                        signal = signal[-target_len:]
                    else:
                        pad = np.zeros(target_len - len(signal))
                        signal = np.concatenate([pad, signal])
                    
                    # MinMax Normalize
                    _min, _max = np.min(signal), np.max(signal)
                    signal_norm = (signal - _min) / (_max - _min + 1e-8)
                    
                    # Reshape for Model
                    X_signal = signal_norm.reshape(1, target_len, 1)
                    X_tabular = np.array([[parity, gestation, age]]) # Order must match training
                    
                    # 4. Predict
                    prediction = model.predict([X_signal, X_tabular])
                    prob = float(prediction[0][0])
                    
                    # 5. Render Dashboard
                    
                    # Row 1: KPI formatting
                    render_kpi_cards(prob, len(signal)/60) # 1Hz assumption for duration
                    
                    st.divider()
                    
                    # Row 2: Diagnosis & Signal
                    c_diag, c_chart = st.columns([1, 2])
                    
                    with c_diag:
                        render_diagnosis(prob)
                        st.markdown("### Clinical Summary")
                        st.dataframe({
                            "Metric": ["Patient Age", "Gestation", "Parity", "Basal Rate"],
                            "Value": [f"{age}", f"{gestation} wk", f"{parity}", "N/A"]
                        }, hide_index=True, width="stretch") # Updated param

                    with c_chart:
                        render_signal_chart(signal_norm, fs=1, theme=theme)
                    
                    st.divider()
                    
                    # Row 3: XAI
                    render_xai(model, X_signal, X_tabular, {'Age': age, 'Gestation': gestation, 'Parity': parity}, theme=theme)
                    
                    # Cleanup using the absolute paths
                    try:
                        os.remove(path_dat)
                        os.remove(path_hea)
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