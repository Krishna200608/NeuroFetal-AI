
import re
import streamlit as st

def parse_header_metadata(header_file):
    """
    Parses Age, Parity, and Gestation from .hea file comments.
    Expected format: 
    # Age: 25   OR   # Age 25
    # Parity: 0 OR   # Parity 0
    # Gestation: 39
    """
    try:
        content = header_file.getvalue().decode("utf-8")
        meta = {}
        
        # Regex for standard PhysioNet header comments
        # Handles optional colon and varying whitespace
        # \s*[:\s]+\s* -> matches ": " or " " or ":"
        age_match = re.search(r'#?\s*Age\s*[:\s]+\s*(\d+)', content, re.IGNORECASE)
        parity_match = re.search(r'#?\s*(?:Parity|Gavidity)\s*[:\s]+\s*(\d+)', content, re.IGNORECASE) # Gravidity sometimes used?
        # User file shows #Parity       0.
        
        gest_match = re.search(r'#?\s*(?:Gestation|Gest\. weeks)\s*[:\s]+\s*(\d+)', content, re.IGNORECASE)
        
        if age_match: meta['age'] = int(age_match.group(1))
        if parity_match: meta['parity'] = int(parity_match.group(1))
        if gest_match: meta['gestation'] = int(gest_match.group(1))
        
        return meta
    except Exception as e:
        print(f"Error parsing header: {e}")
        return {}

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
