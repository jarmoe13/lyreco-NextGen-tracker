import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from transformers import pipeline

# --- CONFIG ---
st.set_page_config(page_title="Lyreco Next Gen Tracker", layout="wide", page_icon="🛒")

# --- STYLE ---
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {font-size: 24px;}
</style>
""", unsafe_allow_html=True)

# --- API SETUP ---
try:
    API_KEY = st.secrets["SERPER_API_KEY"]
except:
    API_KEY = None

# --- AI MODEL ---
@st.cache_resource
def load_model():
    # Multilingual model is crucial for EU markets
    return pipeline("sentiment-analysis", 
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")

try:
    with st.spinner("Initializing Next-Gen AI Radar..."):
        sentiment_pipeline = load_model()
except: pass

# --- LOGIC ---
def map_sentiment(label):
    if 'label_0' in str(label).lower() or 'neg' in str(label).lower(): return 'Negative'
    if 'label_2' in str(label).lower() or 'pos' in str(label).lower(): return 'Positive'
    return 'Neutral'

def analyze_sentiment(df):
    if df.empty: return df
    try:
        results = sentiment_pipeline(df['Title'].tolist(), truncation=True, max_length=512)
        df['sentiment'] = [map_sentiment(r['label']) for r in results]
        df['score'] = [r['score'] for r in results]
    except Exception as e:
        df['sentiment'] = "Neutral"
        df['score'] = 0.0
    return df

def fetch_google_data(query, country, lang, api_key):
    """
    Fetches specific e-commerce data using Google Search API (Serper)
    """
    url = "https://google.serper.dev/search"
    
    payload = json.dumps({
        "q": query,
        "gl": country,   # Geo Location (e.g., 'fr')
        "hl": lang,      # Host Language (e.g., 'fr')
        "num": 15,       # Results per market
        "tbs": "qdr:m6"  # Last 6 months (Platform launch window)
    })
    
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        results = response.json()
        
        parsed_data = []
        if 'organic' in results:
            for r in results['organic']:
                parsed_data.append({
                    'Source': 'Web/Forum',
                    'Title': r.get('title', ''),
                    'Link': r.get('link', '#'),
                    'Snippet': r.get('snippet', ''),
                    'Position': r.get('position', 0)
                })
        return parsed_data
    except:
        return []

# --- TARGET CONFIGURATION ---
# To jest serce aplikacji - precyzyjne zapytania o nową platformę
MARKETS = {
    "France 🇫🇷": {
        "code": "fr", "lang": "fr", 
        "query": "Lyreco nouvelle boutique en ligne avis OR problème"
    },
    "Poland 🇵🇱": {
        "code": "pl", "lang": "pl", 
        "query": "Lyreco nowy sklep online opinie OR logowanie"
    },
    "UK & IE 🇬🇧": {
        "code": "gb", "lang": "en", 
        "query": "Lyreco new webshop launch reviews OR migration"
    },
    "Italy 🇮🇹": {
        "code": "it", "lang": "it", 
        "query": "Lyreco nuovo sito e-commerce problemi OR recensioni"
    },
    "Denmark 🇩🇰": {
        "code": "dk", "lang": "da", 
        "query": "Lyreco ny webshop erfaringer"
    },
    "Benelux 🇧🇪🇳🇱": {
        "code": "be", "lang": "nl", 
        "query": "Lyreco nieuwe webshop problemen"
    }
}

# --- UI LAYOUT ---
st.title("🚀 Lyreco Next Gen: Launch Monitor")
st.markdown("### Tracking the deployment of the new E-Commerce Platform across Europe")

with st.sidebar:
    st.header("Scope Control")
    if not API_KEY:
        st.error("🔑 API Key Missing! Add SERPER_API_KEY to Secrets.")
        API_KEY = st.text_input("Or enter key here:", type="password")
    else:
        st.success("✅ Secure Connection Active")
    
    st.divider()
    
    selected_markets = st.multiselect("Active Markets:", list(MARKETS.keys()), default=["France 🇫🇷", "Poland 🇵🇱", "UK & IE 🇬🇧"])
    
    st.info("Scanning for: Migration issues, New UX feedback, Login errors, Launch announcements.")
    
    run_btn = st.button("🛰️ SCAN ECOSYSTEM", type="primary")

if run_btn and API_KEY:
    all_data = []
    progress = st.progress(0)
    
    col1, col2 = st.columns([3, 1])
    status_log = col1.empty()
    
    for i, market in enumerate(selected_markets):
        config = MARKETS[market]
        status_log.info(f"📡 Pinging {market} (Query: '{config['query']}')...")
        
        # Call API
        market_data = fetch_google_data(config['query'], config['code'], config['lang'], API_KEY)
        
        # Tag data
        for item in market_data:
            item['Market'] = market
            all_data.append(item)
            
        progress.progress((i + 1) / len(selected_markets))
    
    progress.empty()
    status_log.success("Scan Complete.")
    
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Analyze Sentiment
        with st.spinner("Analyzing Feedback Tone..."):
            df = analyze_sentiment(df)

        # --- KPI BOARD ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Mentions Found", len(df))
        
        neg_df = df[df['sentiment'] == 'Negative']
        k2.metric("Critical Issues (Neg)", len(neg_df), delta_color="inverse")
        
        pos_df = df[df['sentiment'] == 'Positive']
        k3.metric("Positive Feedback", len(pos_df))
        
        top_m = df['Market'].mode()[0]
        k4.metric("Most Active Market", top_m)
        
        st.divider()

        # --- DEEP DIVE CHARTS ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Sentiment by Market")
            # Stacked bar chart to see where the problems are
            fig_bar = px.histogram(df, x="Market", color="sentiment", 
                                   color_discrete_map={'Positive':'#00CC96', 'Neutral':'#AB63FA', 'Negative':'#EF553B'},
                                   barmode='group')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.subheader("Global Reception")
            fig_pie = px.pie(df, names='sentiment', color='sentiment', hole=0.5,
                             color_discrete_map={'Positive':'#00CC96', 'Neutral':'#AB63FA', 'Negative':'#EF553B'})
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- CRITICAL FEEDBACK TABLE ---
        st.subheader("🚨 Risk Radar (Negative/Critical Mentions)")
        if not neg_df.empty:
            st.dataframe(
                neg_df[['Market', 'Title', 'Snippet', 'Link']],
                column_config={
                    "Link": st.column_config.LinkColumn("View Source"),
                    "Snippet": "Context Detected"
                },
                use_container_width=True
            )
        else:
            st.success("No critical negative feedback detected regarding the new platform.")

        # --- FULL DATA ---
        with st.expander("View All Data (Raw Log)"):
            st.dataframe(df, use_container_width=True)
            
    else:
        st.warning("No specific data found for 'Next Gen' launch in selected markets. Try extending the date range or simplified queries.")

elif run_btn and not API_KEY:
    st.error("Please provide an API Key to start.")
