import streamlit as st
import pandas as pd
import plotly.express as px
from pygooglenews import GoogleNews
from duckduckgo_search import DDGS
from transformers import pipeline
import time

# --- CONFIG ---
st.set_page_config(page_title="Lyreco Digital & Market Intel", layout="wide", page_icon="📡")

# --- TARGET MARKETS & KEYWORDS (BROADENED) ---
# Teraz słowa są szersze. Jeśli nie znajdzie "launch", znajdzie "digital".
MARKETS = {
    "France 🇫🇷": {
        "geo": "FR", "lang": "fr", 
        "keywords": ["digital", "e-commerce", "commande", "site", "app", "RSE", "logistique"],
        "fallback": True
    },
    "Poland 🇵🇱": {
        "geo": "PL", "lang": "pl", 
        "keywords": ["online", "platforma", "cyfryzacja", "sklep", "aplikacja", "CSR", "magazyn"],
        "fallback": True
    },
    "Denmark 🇩🇰": {
        "geo": "DK", "lang": "da", 
        "keywords": ["digital", "webshop", "online", "bæredygtighed", "app"],
        "fallback": True
    },
    "Italy 🇮🇹": {
        "geo": "IT", "lang": "it", 
        "keywords": ["digitale", "piattaforma", "sito", "sostenibilità", "ecommerce"],
        "fallback": True
    },
    "UK & Ireland 🇬🇧🇮🇪": {
        "geo": "GB", "lang": "en", 
        "keywords": ["digital", "online", "platform", "sustainability", "supply chain"],
        "fallback": True
    }
}

# --- AI SETUP ---
@st.cache_resource
def load_model():
    # Używamy lekkiego modelu, żeby nie zapchać pamięci przy szerszym wyszukiwaniu
    return pipeline("sentiment-analysis", 
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")

try:
    with st.spinner("Loading AI Engines..."):
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
    
    # 1. Google News (Broad Search)
    try:
        gn = GoogleNews(lang=config['lang'], country=config['geo'])
        # Szukamy ogólnie Lyreco, a potem filtrujemy w Pythonie (skuteczniejsze)
        search = gn.search(base_query, when="6m")
        
        for entry in search['entries']:
            # Sprawdzamy czy tytuł zawiera słowa kluczowe LUB czy to fallback
            title_lower = entry.title.lower()
            topic = "General Brand News" # Domyślna kategoria
            
            for kw in config['keywords']:
                if kw in title_lower:
                    topic = f"Topic: {kw.capitalize()}"
                    break
            
            data.append({
                'Market': market_name,
                'Source': 'Google News',
                'Title': entry.title,
                'Date': entry.published,
                'Link': entry.link,
                'Topic': topic
            })
    except: pass

    # 2. DuckDuckGo (LinkedIn & Context)
    try:
        with DDGS() as ddgs:
            # Trick: szukamy na LinkedIn w danym kraju
            q_social = f'site:linkedin.com/company/lyreco "{config["lang"]}"'
            results = list(ddgs.text(q_social, max_results=4))
            for r in results:
                data.append({
                    'Market': market_name,
                    'Source': 'LinkedIn / Social',
                    'Title': r['title'],
                    'Date': None,
                    'Link': r['href'],
                    'Topic': "Digital/Corporate Update"
                })
    except: pass
    
    return data

# --- UI ---
st.title("📡 Lyreco Market Intelligence Radar")
st.markdown("**Scope:** Digital footprint, E-commerce signals & General Brand Activity across Europe.")

with st.sidebar:
    st.header("Scanning Regions")
    selected_markets = []
    for market in MARKETS.keys():
        if st.checkbox(market, value=True):
            selected_markets.append(market)
    
    st.divider()
    run_btn = st.button("🚀 FULL SCAN START", type="primary")

if run_btn:
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, market in enumerate(selected_markets):
        progress_bar.progress((i + 1) / len(selected_markets))
        status_text.text(f"📡 Intercepting signals from {market}...")
        
        market_data = search_market(market, MARKETS[market])
        all_results.extend(market_data)
        
    df = pd.DataFrame(all_results)
    progress_bar.empty()
    status_text.empty()
    
    if df.empty:
        st.error("Total radio silence. This is highly unusual. Check internet connection.")
    else:
        # Data Cleanup
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
        df = analyze_sentiment(df)
        
        # --- DASHBOARD ---
        
        # KPI ROW
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Signals Detected", len(df))
        kpi2.metric("Active Markets", df['Market'].nunique())
        top_topic = df['Topic'].mode()[0] if not df.empty else "N/A"
        kpi3.metric("Top Discussion Topic", top_topic)
        
        st.divider()
        
        # 1. Market Heatmap
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🌍 Activity by Market")
            market_counts = df['Market'].value_counts().reset_index()
            market_counts.columns = ['Market', 'Signals']
            fig_bar = px.bar(market_counts, x='Market', y='Signals', color='Signals', 
                             color_continuous_scale='Blues', text_auto=True)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with c2:
            st.subheader("🤖 Sentiment AI Analysis")
            fig_pie = px.pie(df, names='sentiment', color='sentiment', 
                             color_discrete_map={'Positive':'#00CC96', 'Neutral':'#AB63FA', 'Negative':'#EF553B'})
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # 2. Topic Breakdown
        st.subheader("🔥 What are they talking about?")
        # Odfiltrujemy "General Brand News" żeby zobaczyć konkrety
        specific_topics = df[df['Topic'] != "General Brand News"]
        if not specific_topics.empty:
            fig_topics = px.treemap(specific_topics, path=['Market', 'Topic'], color='sentiment')
            st.plotly_chart(fig_topics, use_container_width=True)
        else:
            st.info("Mostly general news detected. No specific digital launch keywords spiked.")

        # 3. The Feed
        st.subheader("🗞️ Intelligence Feed (Live)")
        
        # Filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            m_filter = st.multiselect("Filter by Market", options=df['Market'].unique(), default=df['Market'].unique())
        with filter_col2:
            s_filter = st.multiselect("Filter by Sentiment", options=df['sentiment'].unique(), default=df['sentiment'].unique())
            
        filtered_df = df[df['Market'].isin(m_filter) & df['sentiment'].isin(s_filter)]
        
        st.dataframe(
            filtered_df[['Date', 'Market', 'Topic', 'Title', 'sentiment', 'Link']],
            column_config={
                "Link": st.column_config.LinkColumn("Source"),
                "Topic": st.column_config.TextColumn("Category"),
            },
            use_container_width=True
        )
