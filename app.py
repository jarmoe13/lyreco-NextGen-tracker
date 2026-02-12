import streamlit as st
import pandas as pd
import plotly.express as px
from pygooglenews import GoogleNews
from duckduckgo_search import DDGS
from transformers import pipeline
import time

# --- CONFIG ---
st.set_page_config(page_title="Lyreco E-Commerce Launch Tracker", layout="wide", page_icon="🛒")

# --- TARGET MARKETS & KEYWORDS ---
# To jest mózg operacji - tłumaczymy frazy na lokalne języki
MARKETS = {
    "France 🇫🇷": {
        "geo": "FR", "lang": "fr", 
        "keywords": ["nouvelle boutique", "nouveau site", "plateforme e-commerce", "digital", "webshop"]
    },
    "Poland 🇵🇱": {
        "geo": "PL", "lang": "pl", 
        "keywords": ["nowy sklep", "nowa platforma", "sklep online", "e-commerce", "cyfryzacja"]
    },
    "Denmark 🇩🇰": {
        "geo": "DK", "lang": "da", # Duński
        "keywords": ["ny webshop", "ny platform", "online butik", "digitalisering", "e-handel"]
    },
    "Italy 🇮🇹": {
        "geo": "IT", "lang": "it", 
        "keywords": ["nuovo sito", "nuova piattaforma", "e-commerce", "digitale", "shop online"]
    },
    "UK & Ireland 🇬🇧🇮🇪": {
        "geo": "GB", "lang": "en", 
        "keywords": ["new webshop", "new platform", "online store", "digital transformation", "e-commerce launch"]
    }
}

# --- AI SETUP ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", 
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")

try:
    with st.spinner("Calibrating AI Radar..."):
        sentiment_pipeline = load_model()
except: st.error("AI Model Error")

# --- LOGIC ---
def map_label(label):
    if 'label_0' in str(label).lower(): return 'Negative'
    if 'label_2' in str(label).lower(): return 'Positive'
    return 'Neutral'

def analyze_sentiment(df):
    if df.empty: return df
    results = sentiment_pipeline(df['title'].tolist(), truncation=True, max_length=512)
    df['sentiment'] = [map_label(r['label']) for r in results]
    df['score'] = [r['score'] for r in results]
    return df

def search_market(market_name, config):
    data = []
    base_query = "Lyreco"
    
    # 1. Google News Specific for Country
    try:
        gn = GoogleNews(lang=config['lang'], country=config['geo'])
        # Szukamy kombinacji: "Lyreco" + "słowo kluczowe"
        for kw in config['keywords']:
            search = gn.search(f"{base_query} {kw}", when="6m") # 6 miesięcy wstecz - to duży projekt
            for entry in search['entries']:
                data.append({
                    'Market': market_name,
                    'Source': 'Google News',
                    'Title': entry.title,
                    'Date': entry.published,
                    'Link': entry.link,
                    'Keyword Match': kw
                })
    except: pass

    # 2. DuckDuckGo (Forums/Blogs)
    try:
        with DDGS() as ddgs:
            for kw in config['keywords']:
                q = f'{base_query} {kw} site:{config["geo"]}' # Wymuszamy domeny krajowe np. site:pl
                results = list(ddgs.text(q, max_results=3))
                for r in results:
                    data.append({
                        'Market': market_name,
                        'Source': 'Web/Social',
                        'Title': r['title'],
                        'Date': None,
                        'Link': r['href'],
                        'Keyword Match': kw
                    })
    except: pass
    
    return data

# --- UI ---
st.title("🛒 Lyreco E-Commerce Launch Tracker")
st.markdown("**Scope:** Monitoring deployment of new digital platforms across key European markets.")

with st.sidebar:
    st.header("Target Markets")
    selected_markets = []
    # Generujemy checkboxy dla każdego kraju
    for market in MARKETS.keys():
        if st.checkbox(market, value=True):
            selected_markets.append(market)
    
    st.divider()
    run_btn = st.button("🛰️ SCAN MARKETS", type="primary")

if run_btn:
    all_results = []
    progress_bar = st.progress(0)
    
    for i, market in enumerate(selected_markets):
        # Update progress
        progress_bar.progress((i + 1) / len(selected_markets))
        st.toast(f"Scanning {market} ecosystem...")
        
        # Search
        market_data = search_market(market, MARKETS[market])
        all_results.extend(market_data)
        
    df = pd.DataFrame(all_results)
    progress_bar.empty()
    
    if df.empty:
        st.warning("No specific e-commerce launch news found in selected markets (last 6 months).")
    else:
        # Data Cleanup
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
        df = analyze_sentiment(df)
        
        # --- DASHBOARD ---
        
        # 1. Market Heatmap
        st.subheader("🌍 Mentions by Market")
        market_counts = df['Market'].value_counts().reset_index()
        market_counts.columns = ['Market', 'Mentions']
        fig_bar = px.bar(market_counts, x='Market', y='Mentions', color='Mentions', 
                         color_continuous_scale='Viridis', text_auto=True)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 2. Key Topics (Keyword breakdown)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment Reception")
            fig_pie = px.pie(df, names='sentiment', color='sentiment', 
                             color_discrete_map={'Positive':'#00CC96', 'Neutral':'#AB63FA', 'Negative':'#EF553B'})
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.subheader("Topic Focus")
            # Co dokładnie wykrył? (Webshop vs Platform vs App)
            fig_topics = px.histogram(df, y='Keyword Match', x='Market', color='sentiment')
            st.plotly_chart(fig_topics, use_container_width=True)
            
        # 3. The Feed
        st.subheader("🗞️ Intelligence Feed")
        for market in selected_markets:
            market_df = df[df['Market'] == market]
            if not market_df.empty:
                with st.expander(f"Show results for {market} ({len(market_df)})", expanded=True):
                    st.dataframe(
                        market_df[['Date', 'Title', 'sentiment', 'Source', 'Link']],
                        column_config={"Link": st.column_config.LinkColumn("Read Article")},
                        use_container_width=True
                    )
