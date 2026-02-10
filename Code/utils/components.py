
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import tensorflow as tf

import base64
import os

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def render_header(theme="Light"):
    # Logo Path (Relative to utils/components.py -> ../assets)
    # User Request: Use Logo-black.png for Dark Mode, Logo.svg for Light Mode
    logo_file = "Logo-black.png" if theme == "Dark" else "Logo.svg"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, "..", "assets", logo_file)
    logo_b64 = get_base64_image(logo_path)
    
    # Determine MIME type based on extension
    mime_type = "image/png" if logo_file.endswith(".png") else "image/svg+xml"
    
    # Icon or Image HTML
    if logo_b64:
        # Dynamic MIME type rendering
        img_html = f'<img src="data:{mime_type};base64,{logo_b64}" style="width: 60px; height: 60px; vertical-align: middle;">'
        icon_container = f"""<div style="display: flex; align-items: center; justify-content: center;">{img_html}</div>"""
    else:
        # Fallback
        icon_container = """<div style="background: #e1f5fe; padding: 10px; border-radius: 12px; color: #0288d1;"><span class="material-symbols-rounded" style="font-size: 32px;">cardiology</span></div>"""

    st.markdown(f"""
<div class="header-container">
    <div style="display: flex; align-items: center; gap: 15px;">
        {icon_container}
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
            <div class="metric-label"><span class="material-symbols-rounded" style="font-size:18px; color:#64748b; margin-right:5px;">timer</span> Duration</div>
            <div class="metric-value text-blue">{signal_dur_min:.1f} m</div>
            <div class="metric-delta">Signal Length</div>
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

def render_mismatch_error(dat_name, hea_name, theme="Light"):
    # Theme colors
    bg_color = "#fef2f2" if theme == "Light" else "#2b1515"
    border_color = "#fecaca" if theme == "Light" else "#451a1a"
    text_color = "#991b1b" if theme == "Light" else "#fecaca"
    subtext_color = "#b91c1c" if theme == "Light" else "#fca5a5"
    
    st.markdown(f"""
<div style="text-align: center; padding: 40px; background-color: {bg_color}; border: 2px dashed {border_color}; border-radius: 16px; max-width: 700px; margin: 20px auto;">
<span class="material-symbols-rounded" style="font-size: 56px; color: #ef4444; margin-bottom: 15px; display: block;">folder_off</span>
<h2 style="color: {text_color}; margin: 0 0 10px 0; font-size: 1.8rem;">File Mismatch Detected</h2>
<p style="color: {subtext_color}; font-size: 1.1rem; margin-bottom: 25px;">The uploaded files do not identify to the same PhysioNet record.</p>
<div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; margin-bottom: 25px;">
<div style="text-align: right; flex: 1;">
<span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: {subtext_color}; opacity: 0.8;">Signal File</span>
<div style="font-family: monospace; font-size: 1.2rem; font-weight: 600; color: {text_color}; background: rgba(255,0,0,0.05); padding: 8px 12px; border-radius: 8px; border: 1px solid {border_color};">{dat_name}</div>
</div>
<div style="display: flex; align-items: center; height: 100%; padding-top: 20px;">
<span class="material-symbols-rounded" style="color: #ef4444; font-size: 32px;">link_off</span>
</div>
<div style="text-align: left; flex: 1;">
<span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: {subtext_color}; opacity: 0.8;">Header File</span>
<div style="font-family: monospace; font-size: 1.2rem; font-weight: 600; color: {text_color}; background: rgba(255,0,0,0.05); padding: 8px 12px; border-radius: 8px; border: 1px solid {border_color};">{hea_name}</div>
</div>
</div>
<div style="background: rgba(239, 68, 68, 0.1); display: inline-block; padding: 8px 16px; border-radius: 20px; color: {text_color}; font-weight: 500;">
<span class="material-symbols-rounded" style="font-size: 16px; vertical-align: text-bottom; margin-right: 5px;">info</span>
Please ensure both files share the same record ID (e.g., <b>1001.dat</b> and <b>1001.hea</b>)
</div>
</div>
""", unsafe_allow_html=True)

def render_uncertainty_analysis(report_dir):
    st.subheader("Model Validation & Uncertainty Analysis (Static)")
    st.markdown("Metrics from the latest model evaluation run (Fold 1).")
    
    col1, col2 = st.columns(2)
    
    # Check robust paths
    cal_path = os.path.join(report_dir, "calibration_curve.png")
    hist_path = os.path.join(report_dir, "uncertainty_histogram.png")
    
    with col1:
        if os.path.exists(cal_path):
            st.image(cal_path, caption="Calibration Curve (Reliability Diagram)", width="stretch")
        else:
            st.warning(f"Calibration plot not found at {cal_path}")
            
    with col2:
        if os.path.exists(hist_path):
            st.image(hist_path, caption="Uncertainty Distribution Histogram", width="stretch")
        else:
            st.warning(f"Uncertainty histogram not found at {hist_path}")
            
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 10px; border-radius: 8px; border: 1px solid #bbdefb; color: #0d47a1;">
        <span class="material-symbols-rounded" style="font-size: 18px; vertical-align: text-bottom; margin-right: 5px;">info</span>
        <b>Interpretation</b>: A well-calibrated model (diagonal line) means predicted probabilities match observed accuracy. The histogram shows how often the model is 'unsure'.
    </div>
    """, unsafe_allow_html=True)
