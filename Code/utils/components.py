
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

def render_header():
    # Logo Path (Relative to app.py execution)
    logo_path = os.path.join("assets", "logo.jpg")
    logo_b64 = get_base64_image(logo_path)
    
    # Icon or Image HTML
    if logo_b64:
        # Rounded image matching the previous container style
        img_html = f'<img src="data:image/jpg;base64,{logo_b64}" style="width: 55px; height: 55px; border-radius: 12px; object-fit: cover;">'
        icon_container = f"""
            <div style="border-radius: 12px; overflow: hidden; display: flex; align-items: center;">
                {img_html}
            </div>
        """
    else:
        # Fallback
        icon_container = """
            <div style="background: #e1f5fe; padding: 10px; border-radius: 12px; color: #0288d1;">
                <span class="material-symbols-rounded" style="font-size: 32px;">cardiology</span>
            </div>
        """

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
