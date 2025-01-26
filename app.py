import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import io
import cv2
import tensorflow as tf
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import yaml
import os

# Importer moduler
from rental_analyzer import RentalAnalyzer, BuildingType
from regulations_handler import RegulationsHandler
from ai_modules.floor_plan_analyzer import FloorPlanAnalyzer
from ai_modules.economic_analyzer import EconomicAnalyzer
from ai_modules.visualization_engine import VisualizationEngine
from ai_modules.support_chatbot import SupportChatbot
from ai_modules.document_generator import DocumentGenerator

# Last konfigurasjon
try:
    with open("config/settings.py") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.warning(f"Kunne ikke laste konfigurasjon: {str(e)}")
    config = {}

# Konfigurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Konfigurasjon for sideoppsett
st.set_page_config(
    page_title="EiendomsAI Pro",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialiser komponenter
@st.cache_resource
def initialize_components():
    """Initialiser alle hovedkomponenter"""
    try:
        components = {
            "rental_analyzer": RentalAnalyzer(),
            "regulations_handler": RegulationsHandler(),
            "floor_plan_analyzer": FloorPlanAnalyzer(),
            "economic_analyzer": EconomicAnalyzer(),
            "visualization_engine": VisualizationEngine(),
            "support_chatbot": SupportChatbot(),
            "document_generator": DocumentGenerator()
        }
        
        # Last AI-modeller
        components.update(load_ai_models())
        
        return components
    except Exception as e:
        logger.error(f"Feil ved initialisering av komponenter: {str(e)}")
        st.error("Kunne ikke initialisere alle komponenter. Noen funksjoner kan være utilgjengelige.")
        return {}

# Sett opp cache for bedre ytelse
@st.cache_data
def load_ai_models():
    """Last inn AI modeller for bildeanalyse og planløsning"""
    try:
        models = {
            "floor_plan_model": tf.keras.models.load_model(
                config.get("AI_MODELS", {}).get("floor_plan", {}).get("model_path", "models/floor_plan_analyzer.h5")
            ),
            "price_estimation_model": tf.keras.models.load_model(
                config.get("AI_MODELS", {}).get("price_estimation", {}).get("model_path", "models/price_estimator.h5")
            ),
            "image_recognition_model": tf.keras.models.load_model(
                config.get("AI_MODELS", {}).get("image_recognition", {}).get("model_path", "models/image_recognizer.h5")
            )
        }
        return models
    except Exception as e:
        logger.error(f"Feil ved lasting av AI-modeller: {str(e)}")
        return {
            "floor_plan_model": None,
            "price_estimation_model": None,
            "image_recognition_model": None
        }

# Stil og tema
st.markdown("""
    <style>
    /* Hovedstil */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Overskrifter */
    .css-10trblm {
        color: #ffffff;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(0,255,255,0.3);
    }
    
    /* Undertitler */
    .css-1vbkxwb {
        color: #4dd0e1;
    }
    
    /* Knapper */
    .stButton>button {
        background: linear-gradient(90deg, #00b4db 0%, #0083b0 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.7rem 1.5rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0,180,219,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,180,219,0.3);
        background: linear-gradient(90deg, #00b4db 20%, #0083b0 100%);
    }
    
    /* Input felter */
    .stTextInput>div>div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        color: white;
    }
    
    /* Selectbox */
    .stSelectbox>div>div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        color: white;
    }
    
    /* AI-elementer */
    .ai-container {
        background: rgba(13,17,23,0.4);
        border: 1px solid rgba(77,208,225,0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .ai-badge {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-bottom: 1rem;
        display: inline-block;
    }
    
    /* Statusbokser */
    .status-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(5px);
    }
    
    .success-box {
        border-left: 4px solid #00f2fe;
    }
    
    .warning-box {
        border-left: 4px solid #ffd700;
    }
    
    .error-box {
        border-left: 4px solid #ff4757;
    }
    
    /* Grafer og visualiseringer */
    .js-plotly-plot {
        background: rgba(13,17,23,0.4);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(77,208,225,0.2);
    }
    
    /* Animerte elementer */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(77,208,225,0.4); }
        70% { box-shadow: 0 0 0 10px rgba(77,208,225,0); }
        100% { box-shadow: 0 0 0 0 rgba(77,208,225,0); }
    }
    
    .ai-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Moderne kort-design */
    .modern-card {
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(13,17,23,0.9);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white;
        border-radius: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00b4db 0%, #0083b0 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialiser komponenter
try:
    components = initialize_components()
    rental_analyzer = components["rental_analyzer"]
    regulations_handler = components["regulations_handler"]
    floor_plan_analyzer = components["floor_plan_analyzer"]
    economic_analyzer = components["economic_analyzer"]
    visualization_engine = components["visualization_engine"]
    support_chatbot = components["support_chatbot"]
    document_generator = components["document_generator"]
    ai_models = components.get("ai_models", {})
except Exception as e:
    logger.error(f"Kritisk feil ved initialisering: {str(e)}")
    st.error("Det oppstod en kritisk feil ved oppstart av applikasjonen. Vennligst prøv igjen senere.")

def main():
    st.title("🏢 EiendomsAI Pro - Intelligent Eiendomsanalyse")
    
    # Sidepanel for navigasjon
    st.sidebar.title("Navigasjon")
    page = st.sidebar.selectbox(
        "Velg funksjon",
        ["Hjem", "Eiendomsanalyse", "3D Visualisering", "Økonomisk Analyse", 
         "Dokumentgenerator", "Kundeservice"]
    )
    
    if page == "Hjem":
        show_home_page()
    elif page == "Eiendomsanalyse":
        show_property_analysis()
    elif page == "3D Visualisering":
        show_3d_visualization()
    elif page == "Økonomisk Analyse":
        show_economic_analysis()
    elif page == "Dokumentgenerator":
        show_document_generator()
    elif page == "Kundeservice":
        show_customer_service()

def show_home_page():
    st.header("Velkommen til EiendomsAI Pro")
    st.write("""
    ### Din intelligente partner for eiendomsutvikling
    
    EiendomsAI Pro er en state-of-the-art plattform som kombinerer:
    - 🤖 Avansert AI og maskinlæring
    - 📊 Presis teknisk analyse
    - 🎯 Automatiserte prosesser
    - 🏗️ 3D visualisering
    - 💰 Detaljert økonomisk analyse
    - 📝 Automatisk dokumentgenerering
    """)
    
    # Vis nøkkeltall og statistikk
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Gjennomsnittlig ROI", value="15.2%", delta="2.1%")
    with col2:
        st.metric(label="Suksessrate", value="94%", delta="3.5%")
    with col3:
        st.metric(label="Behandlingstid", value="3.2 dager", delta="-1.1 dager")

def show_property_analysis():
    st.header("Eiendomsanalyse")
    
    # Last opp dokumenter
    uploaded_file = st.file_uploader("Last opp plantegning eller bilder", 
                                   type=["jpg", "png", "pdf"])
    
    # Finn.no integrasjon
    finn_url = st.text_input("Eller lim inn Finn.no annonselenke")
    
    if uploaded_file is not None or finn_url:
        st.info("Analyserer eiendom... Vennligst vent")
        
        # TODO: Implementer analyse
        
        # Vis resultater i faner
        tab1, tab2, tab3 = st.tabs(["Teknisk Analyse", "Reguleringer", "Anbefalinger"])
        
        with tab1:
            st.subheader("Teknisk Analyse")
            # TODO: Vis teknisk analyse
            
        with tab2:
            st.subheader("Reguleringsplan og Krav")
            # TODO: Vis reguleringsinfo
            
        with tab3:
            st.subheader("Anbefalte Tiltak")
            # TODO: Vis anbefalinger

def show_3d_visualization():
    st.header("3D Visualisering")
    
    # Velg visualiseringstype
    viz_type = st.radio(
        "Velg visualiseringstype",
        ["Nåværende tilstand", "Foreslåtte endringer", "Før/etter sammenligning"]
    )
    
    # Visualiseringsinnstillinger
    st.sidebar.subheader("Innstillinger")
    quality = st.sidebar.select_slider("Kvalitet", 
                                     options=["Lav", "Medium", "Høy"])
    show_measurements = st.sidebar.checkbox("Vis målinger", value=True)
    
    # TODO: Implementer 3D-visualisering
    st.info("3D-modell lastes... Vennligst vent")

def show_economic_analysis():
    st.header("Økonomisk Analyse")
    
    # Input-parametere
    col1, col2 = st.columns(2)
    with col1:
        investment = st.number_input("Initialinvestering (NOK)", 
                                   min_value=0, value=1000000)
        monthly_rent = st.number_input("Forventet månedlig leieinntekt (NOK)", 
                                     min_value=0, value=15000)
        renovation_cost = st.number_input("Estimerte oppussing/tilpasningskostnader (NOK)", 
                                        min_value=0, value=200000)
    with col2:
        loan_ratio = st.slider("Belåningsgrad (%)", 0, 100, 75)
        interest_rate = st.slider("Lånerente (%)", 0.0, 10.0, 4.5)
        loan_years = st.slider("Nedbetalingstid (år)", 1, 30, 20)
    
    # Avanserte innstillinger
    with st.expander("Avanserte innstillinger"):
        col3, col4 = st.columns(2)
        with col3:
            maintenance_cost = st.number_input(
                "Årlig vedlikeholdskostnad (% av eiendomsverdi)",
                min_value=0.0, max_value=10.0, value=1.0, step=0.1
            )
            property_tax = st.number_input(
                "Eiendomsskatt (%)",
                min_value=0.0, max_value=5.0, value=0.3, step=0.1
            )
        with col4:
            inflation_rate = st.number_input(
                "Forventet inflasjon (%)",
                min_value=0.0, max_value=10.0, value=2.5, step=0.1
            )
            rent_increase = st.number_input(
                "Årlig leieprisøkning (%)",
                min_value=0.0, max_value=10.0, value=3.0, step=0.1
            )
    
    if st.button("Utfør analyse", type="primary"):
        with st.spinner("Utfører omfattende økonomisk analyse..."):
            # Samle data
            analysis_data = {
                "investment": investment,
                "monthly_rent": monthly_rent,
                "renovation_cost": renovation_cost,
                "loan_ratio": loan_ratio / 100,
                "interest_rate": interest_rate / 100,
                "loan_years": loan_years,
                "maintenance_cost": maintenance_cost / 100,
                "property_tax": property_tax / 100,
                "inflation_rate": inflation_rate / 100,
                "rent_increase": rent_increase / 100
            }
            
            try:
                # Utfør analyse
                results = economic_analyzer.analyze_investment(analysis_data)
                
                if results:
                    # Vis hovedresultater
                    st.subheader("Hovedresultater")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric(
                            label="Total avkastning (ROI)",
                            value=f"{results['roi']:.1f}%",
                            delta=f"{results['roi'] - 7.0:.1f}%"
                        )
                    
                    with metrics_col2:
                        st.metric(
                            label="Netto årlig inntekt",
                            value=f"{results['annual_net_income']:,.0f} NOK",
                            delta=f"{results['cash_flow_growth']:+.1f}%"
                        )
                    
                    with metrics_col3:
                        st.metric(
                            label="Tilbakebetalingstid",
                            value=f"{results['payback_period']:.1f} år",
                            delta=f"{10 - results['payback_period']:.1f} år",
                            delta_color="inverse"
                        )
                    
                    # Vis detaljert analyse i faner
                    tab1, tab2, tab3 = st.tabs([
                        "Kontantstrømanalyse",
                        "Lønnsomhetsberegning",
                        "Risikovurdering"
                    ])
                    
                    with tab1:
                        st.subheader("Kontantstrømanalyse")
                        # Vis kontantstrømgraf
                        fig_cash_flow = create_cash_flow_chart(results['cash_flow_data'])
                        st.plotly_chart(fig_cash_flow, use_container_width=True)
                        
                    with tab2:
                        st.subheader("Lønnsomhetsberegning")
                        col5, col6 = st.columns(2)
                        
                        with col5:
                            st.write("### Inntekter")
                            st.write(f"""
                            - Årlig leieinntekt: {results['annual_rent_income']:,.0f} NOK
                            - Verdiøkning: {results['property_value_increase']:,.0f} NOK
                            - Total inntekt: {results['total_income']:,.0f} NOK
                            """)
                            
                        with col6:
                            st.write("### Utgifter")
                            st.write(f"""
                            - Lånnekostnader: {results['loan_costs']:,.0f} NOK
                            - Vedlikehold: {results['maintenance_costs']:,.0f} NOK
                            - Skatter og avgifter: {results['tax_costs']:,.0f} NOK
                            """)
                            
                    with tab3:
                        st.subheader("Risikovurdering")
                        # Vis risikomatrise
                        risk_data = create_risk_matrix(results['risk_assessment'])
                        st.plotly_chart(risk_data, use_container_width=True)
                        
                    # Vis anbefalinger
                    st.subheader("Anbefalinger")
                    for recommendation in results['recommendations']:
                        st.info(recommendation)
                        
            except Exception as e:
                st.error(f"En feil oppstod under analysen: {str(e)}")
                
def create_cash_flow_chart(data):
    """Lag kontantstrømgraf"""
    fig = go.Figure()
    
    # Legg til kontantstrømlinjer
    fig.add_trace(go.Scatter(
        x=data['years'],
        y=data['cumulative_cash_flow'],
        name="Akkumulert kontantstrøm",
        line=dict(color='#2ecc71', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['years'],
        y=data['net_income'],
        name="Årlig nettoinntekt",
        line=dict(color='#3498db', width=2)
    ))
    
    # Oppdater layout
    fig.update_layout(
        title="Kontantstrømanalyse over tid",
        xaxis_title="År",
        yaxis_title="NOK",
        hovermode='x unified'
    )
    
    return fig
    
def create_risk_matrix(risk_data):
    """Lag risikomatrise"""
    # Opprett risikomatrise med Plotly
    fig = go.Figure()
    
    # Legg til risikopunkter
    fig.add_trace(go.Scatter(
        x=risk_data['probability'],
        y=risk_data['impact'],
        mode='markers+text',
        name='Risikofaktorer',
        text=risk_data['factors'],
        textposition="top center",
        marker=dict(
            size=12,
            color=risk_data['risk_scores'],
            colorscale='RdYlGn_r',
            showscale=True
        )
    ))
    
    # Oppdater layout
    fig.update_layout(
        title="Risikomatrise",
        xaxis_title="Sannsynlighet",
        yaxis_title="Konsekvens",
        showlegend=False
    )
    
    return fig

def show_document_generator():
    st.header("Dokumentgenerator")
    
    # Velg dokumenttype
    doc_type = st.selectbox(
        "Velg dokumenttype",
        ["Søknad om bruksendring", "Teknisk rapport", 
         "Økonomisk analyse", "Komplett søknadspakke"]
    )
    
    # Opprett kolonner for inndata
    col1, col2 = st.columns(2)
    
    with col1:
        # Grunnleggende informasjon
        st.subheader("Grunnleggende informasjon")
        property_address = st.text_input("Eiendomsadresse")
        property_id = st.text_input("Eiendoms-ID / Gårds- og bruksnummer")
        owner_name = st.text_input("Eier's navn")
        owner_contact = st.text_input("Kontaktinformasjon")
        
    with col2:
        # Dokumentspesifikke innstillinger
        st.subheader("Dokumentinnstillinger")
        
        if doc_type == "Søknad om bruksendring":
            current_use = st.selectbox(
                "Nåværende bruk",
                ["Bolig", "Næring", "Kombinert", "Annet"]
            )
            planned_use = st.selectbox(
                "Planlagt bruk",
                ["Utleieenhet", "Hjemmekontor", "Kombinert bolig/næring", "Annet"]
            )
            area_affected = st.number_input(
                "Berørt areal (m²)",
                min_value=0.0,
                step=0.1
            )
            
        elif doc_type == "Teknisk rapport":
            inspection_date = st.date_input("Inspeksjonsdato")
            inspector_name = st.text_input("Inspektør's navn")
            inspection_type = st.selectbox(
                "Type inspeksjon",
                ["Tilstandsrapport", "Bygningsteknisk", "Brann og sikkerhet", "Komplett"]
            )
            
        elif doc_type == "Økonomisk analyse":
            analysis_period = st.slider(
                "Analyseperiode (år)",
                min_value=1,
                max_value=30,
                value=10
            )
            include_scenarios = st.checkbox(
                "Inkluder scenarioanalyse",
                value=True
            )
            
    # Avanserte innstillinger
    with st.expander("Avanserte innstillinger"):
        col3, col4 = st.columns(2)
        with col3:
            document_language = st.selectbox(
                "Språk",
                ["Norsk (Bokmål)", "Norsk (Nynorsk)", "Engelsk"]
            )
            document_format = st.selectbox(
                "Dokumentformat",
                ["PDF", "Word", "Begge"]
            )
        with col4:
            include_attachments = st.checkbox(
                "Inkluder vedlegg",
                value=True
            )
            include_drawings = st.checkbox(
                "Inkluder tegninger",
                value=True
            )
    
    # Last opp tilleggsdokumenter
    st.subheader("Tilleggsdokumenter")
    uploaded_files = st.file_uploader(
        "Last opp relevante dokumenter",
        accept_multiple_files=True,
        type=["pdf", "doc", "docx", "jpg", "png"]
    )
    
    # Generer dokument
    if st.button("Generer dokument", type="primary"):
        with st.spinner("Genererer omfattende dokumentpakke..."):
            try:
                # Samle all informasjon
                document_data = {
                    "type": doc_type,
                    "property": {
                        "address": property_address,
                        "id": property_id
                    },
                    "owner": {
                        "name": owner_name,
                        "contact": owner_contact
                    },
                    "settings": {
                        "language": document_language,
                        "format": document_format,
                        "include_attachments": include_attachments,
                        "include_drawings": include_drawings
                    }
                }
                
                # Legg til dokumenttype-spesifikk informasjon
                if doc_type == "Søknad om bruksendring":
                    document_data.update({
                        "current_use": current_use,
                        "planned_use": planned_use,
                        "area_affected": area_affected
                    })
                elif doc_type == "Teknisk rapport":
                    document_data.update({
                        "inspection_date": inspection_date.strftime("%Y-%m-%d"),
                        "inspector_name": inspector_name,
                        "inspection_type": inspection_type
                    })
                elif doc_type == "Økonomisk analyse":
                    document_data.update({
                        "analysis_period": analysis_period,
                        "include_scenarios": include_scenarios
                    })
                
                # Generer dokumenter
                generated_docs = document_generator.generate_document(
                    document_data,
                    uploaded_files
                )
                
                if generated_docs:
                    st.success("Dokumenter er generert!")
                    
                    # Vis preview og nedlastingslinker
                    st.subheader("Genererte dokumenter")
                    for doc in generated_docs:
                        col5, col6 = st.columns([3, 1])
                        with col5:
                            st.write(f"📄 {doc['name']}")
                        with col6:
                            st.download_button(
                                label="Last ned",
                                data=doc['content'],
                                file_name=doc['name'],
                                mime=doc['mime_type']
                            )
                            
                    # Vis sammendrag
                    st.subheader("Sammendrag")
                    st.write(f"""
                    - Dokumenttype: {doc_type}
                    - Antall sider: {sum(doc['pages'] for doc in generated_docs)}
                    - Inkluderte vedlegg: {len(uploaded_files)}
                    - Generert: {datetime.now().strftime('%d.%m.%Y %H:%M')}
                    """)
                    
                    # Vis neste steg
                    st.info("""
                    ### Neste steg
                    1. Gjennomgå dokumentene nøye
                    2. Signer der det er påkrevet
                    3. Send inn til relevant myndighet
                    """)
                    
            except Exception as e:
                st.error(f"En feil oppstod under dokumentgenerering: {str(e)}")

def show_customer_service():
    st.header("Kundeservice AI")
    
    # Initialiser chat-historikk i session state hvis den ikke eksisterer
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Hei! Jeg er din digitale assistent for EiendomsAI Pro. " + 
                      "Jeg kan hjelpe deg med spørsmål om:\n" +
                      "- Tekniske krav og reguleringer\n" +
                      "- Økonomiske analyser og beregninger\n" +
                      "- Søknadsprosesser og dokumentasjon\n" +
                      "- Generell veiledning om utleie\n\n" +
                      "Hvordan kan jeg hjelpe deg i dag?"
        })
    
    # Vis chat-historikk
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Vis eventuelle vedlegg eller visualiseringer
            if "attachments" in message:
                for attachment in message["attachments"]:
                    if attachment["type"] == "image":
                        st.image(attachment["content"], caption=attachment["caption"])
                    elif attachment["type"] == "chart":
                        st.plotly_chart(attachment["content"])
                    elif attachment["type"] == "file":
                        st.download_button(
                            label=f"Last ned {attachment['name']}",
                            data=attachment["content"],
                            file_name=attachment["name"]
                        )
    
    # Brukerinndata
    user_question = st.chat_input("Skriv din melding her...")
    
    if user_question:
        # Legg til brukerens spørsmål i historikken
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Vis brukerens spørsmål
        with st.chat_message("user"):
            st.write(user_question)
        
        # Behandle spørsmålet og generer svar
        try:
            with st.spinner("Analyserer spørsmålet..."):
                # Analyser spørsmålet og generer svar
                response = support_chatbot.process_question(user_question)
                
                # Forbered vedlegg hvis relevant
                attachments = []
                
                # Sjekk om spørsmålet handler om reguleringer
                if "regul" in user_question.lower() or "krav" in user_question.lower():
                    # Legg ved relevant reguleringsinfo
                    regulation_info = regulations_handler.get_relevant_regulations(user_question)
                    if regulation_info:
                        attachments.append({
                            "type": "file",
                            "name": "reguleringsinfo.pdf",
                            "content": regulation_info
                        })
                
                # Sjekk om spørsmålet handler om økonomi
                if any(word in user_question.lower() for word in ["økonomi", "kostnad", "pris", "lån"]):
                    # Generer relevant økonomisk visualisering
                    chart = economic_analyzer.create_relevant_visualization(user_question)
                    if chart:
                        attachments.append({
                            "type": "chart",
                            "content": chart
                        })
                
                # Legg til svar i historikken
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "attachments": attachments if attachments else None
                })
                
                # Vis svaret
                with st.chat_message("assistant"):
                    st.write(response)
                    
                    # Vis eventuelle vedlegg
                    if attachments:
                        for attachment in attachments:
                            if attachment["type"] == "chart":
                                st.plotly_chart(attachment["content"])
                            elif attachment["type"] == "file":
                                st.download_button(
                                    label=f"Last ned {attachment['name']}",
                                    data=attachment["content"],
                                    file_name=attachment["name"]
                                )
                
                # Vis relaterte spørsmål
                with st.expander("Relaterte spørsmål"):
                    related_questions = support_chatbot.get_related_questions(user_question)
                    for q in related_questions:
                        if st.button(q):
                            # Simuler at brukeren stilte dette spørsmålet
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": q
                            })
                            # Generer svar (vil vises ved neste refresh)
                
        except Exception as e:
            st.error(f"Beklager, jeg kunne ikke prosessere spørsmålet ditt: {str(e)}")
            
    # Vis kontrollpanel for samtalen
    with st.sidebar:
        st.subheader("Samtaleinnstillinger")
        
        # Mulighet for å laste ned chat-historikk
        if st.button("Last ned samtalehistorikk"):
            # Konverter chat-historikk til tekstformat
            chat_text = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in st.session_state.chat_history
            ])
            
            # Tilby nedlasting
            st.download_button(
                label="Last ned som tekstfil",
                data=chat_text,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        # Mulighet for å starte ny samtale
        if st.button("Start ny samtale"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
