"""
Interface Streamlit pour le système de recommandation
Permet de tester les 3 méthodes : Content-Based (avec PCA), Collaborative Filtering, et Popularity
Avec gestion complète du Cold Start
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import requests
import os
from pathlib import Path

# =====================================================
# CONFIGURATION CLOUD RUN API
# =====================================================

# L'URL sera définie via variable d'environnement ou config Streamlit
API_URL = os.getenv("API_URL", "https://recommendation-api-41998741998.europe-west1.run.app")

def call_api_recommend(user_id, method="content", n=5):
    """Appelle l'endpoint /recommend de l'API Cloud Run"""
    try:
        url = f"{API_URL}/recommend/{user_id}"
        params = {"method": method, "n": n}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"error": "Timeout - L'API met trop de temps à répondre"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur de connexion: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

def call_api_health():
    """Vérifie l'état de l'API"""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        return None

def call_api_users(limit=20):
    """Récupère la liste des utilisateurs actifs"""
    try:
        resp = requests.get(f"{API_URL}/users", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        return None

def call_api_popular(n=10):
    """Récupère les articles populaires"""
    try:
        resp = requests.get(f"{API_URL}/popular", params={"n": n}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        return None

# Configuration de la page
st.set_page_config(
    page_title="My Content - Système de Recommandation",
    page_icon="📚",
    layout="wide"
)

# =====================================================
# 2. FONCTIONS D'AFFICHAGE
# =====================================================

def display_recommendations(recommendations, method_name):
    """Affiche les recommandations sous forme de cards"""
    
    st.subheader(f"📚 {method_name}")
    
    if not recommendations:
        st.warning("Aucune recommandation disponible")
        return
    
    for i, rec in enumerate(recommendations, 1):
        # Support pour les deux formats: tuple (article_id, score) ou dict
        if isinstance(rec, tuple):
            article_id, score = rec
            category = "N/A"
            words = "N/A"
        else:
            # Format dict de l'API
            article_id = rec.get('article_id')
            score = rec.get('score', 0)
            category = rec.get('category_id', 'N/A')
            words = rec.get('words_count', 'N/A')
        
        with st.container():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.markdown(f"### #{i}")
            
            with col2:
                st.markdown(f"**Article {article_id}**")
                st.markdown(f"📊 Score: `{score:.4f}` | 🏷️ Catégorie: `{category}` | 📝 Mots: `{words}`")
            
            st.markdown("---")

# =====================================================
# 3. INTERFACE PRINCIPALE
# =====================================================

def main():
    # En-tête
    st.title("📚 My Content - Système de Recommandation")
    st.markdown("""
    Application de démonstration pour tester 3 approches de recommandation :
    - **Content-Based Filtering** (embeddings + PCA 100D)
    - **Collaborative Filtering** (SVD)
    - **Popularity** (fallback cold start)
    """)
    
    # Vérifier si l'API est accessible
    with st.spinner("🔍 Connexion à l'API..."):
        api_status = call_api_health()
    
    if api_status:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Utilisateurs", f"{api_status['users_count']:,}")
        with col2:
            st.metric("📄 Articles", f"{api_status['articles_count']:,}")
        with col3:
            st.metric("🔗 Interactions", f"{api_status['interactions_count']:,}")
        st.success(f"✅ API connectée : {API_URL}")
    else:
        st.error(f"❌ API non accessible : {API_URL}")
        st.info("💡 Vérifiez que votre API FastAPI est déployée et accessible")
        st.code(f"API_URL = {API_URL}", language="text")
        st.stop()
    
    st.markdown("---")
    
    # =====================================================
    # 4. SIDEBAR - SELECTION UTILISATEUR
    # =====================================================
    
    st.sidebar.header("⚙️ Configuration")
    
    # Sélection utilisateur
    st.sidebar.subheader("👤 Sélection utilisateur")
    
    # Récupérer les users depuis l'API
    users_data = call_api_users(limit=30)
    
    if users_data and 'users' in users_data:
        top_users = [u['user_id'] for u in users_data['users']]
        
        user_selection_mode = st.sidebar.radio(
            "Mode de sélection",
            ["Top utilisateurs actifs", "ID manuel"]
        )
        
        if user_selection_mode == "Top utilisateurs actifs":
            selected_user = st.sidebar.selectbox(
                "Utilisateur",
                top_users,
                format_func=lambda x: f"User {x}"
            )
        else:
            selected_user = st.sidebar.number_input(
                "ID utilisateur",
                min_value=1,
                value=top_users[0] if top_users else 1
            )
    else:
        st.sidebar.warning("⚠️ Impossible de charger les utilisateurs")
        selected_user = st.sidebar.number_input("ID utilisateur", min_value=1, value=1)
    
    # Méthode de recommandation
    st.sidebar.subheader("🎯 Méthode de recommandation")
    
    method = st.sidebar.radio(
        "Choisir la méthode",
        ["Content-Based", "Collaborative", "Popularity", "Comparaison des 3"]
    )
    
    # Nombre de recommandations
    n_recs = st.sidebar.slider(
        "Nombre de recommandations",
        min_value=3,
        max_value=10,
        value=5
    )
    
    generate_button = st.sidebar.button("🚀 Générer les recommandations", type="primary")
    
    # =====================================================
    # 5. AFFICHAGE DES RECOMMANDATIONS
    # =====================================================
    
    if generate_button:
        
        st.header(f"👤 Utilisateur {selected_user}")
        st.info(f"🔍 Recherche de recommandations pour l'utilisateur {selected_user}")
        st.markdown("---")
        
        if method == "Comparaison des 3":
            st.header("⚖️ Comparaison des 3 méthodes")
            
            col1, col2, col3 = st.columns(3)
            
            methods = [
                ("content", "Content-Based", col1),
                ("collaborative", "Collaborative", col2),
                ("popularity", "Popularity", col3)
            ]
            
            for api_method, name, col in methods:
                with col:
                    with st.spinner(f"⏳ {name}..."):
                        result = call_api_recommend(selected_user, api_method, n_recs)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        recs = result.get("recommendations", [])
                        cold_start = result.get("cold_start_applied", False)
                        if cold_start:
                            st.warning("⚠️ Cold Start détecté")
                        display_recommendations(recs, name)
        
        else:
            # Méthode unique
            api_method = method.lower().replace("-", "").replace(" ", "")
            
            st.subheader(f"📌 Recommandations via {method}")
            
            with st.spinner("⏳ Appel à l'API Cloud Run..."):
                result = call_api_recommend(selected_user, api_method, n_recs)
            
            if "error" in result:
                st.error(f"❌ Erreur API : {result['error']}")
            else:
                recs = result.get("recommendations", [])
                cold_start = result.get("cold_start_applied", False)
                
                if cold_start:
                    st.warning("⚠️ Cold Start détecté - Recommandations basées sur la popularité")
                
                display_recommendations(recs, method)
    
    else:
        st.info("👈 Configurez les paramètres dans la sidebar et cliquez sur 'Générer les recommandations'")
        
        # Afficher les articles populaires en attendant
        st.subheader("🔥 Articles les plus populaires")
        
        with st.spinner("Chargement..."):
            popular = call_api_popular(10)
        
        if popular and 'popular_articles' in popular:
            display_recommendations(popular['popular_articles'], "Top 10 Articles")
    
    # =====================================================
    # FOOTER
    # =====================================================
    
    st.markdown("---")
    st.markdown(f"""
    ### 💡 À propos
    
    **Projet 10 - OpenClassrooms AI Engineer**
    
    - Dataset : Globo.com (Kaggle)
    - 3 méthodes : Content-Based (PCA 100D), Collaborative (SVD), Popularity
    - API déployée sur Google Cloud Run
    - Cold Start entièrement géré
    
    🔗 API Backend : `{API_URL}`
    """)


# =====================================================
# LANCEMENT
# =====================================================

if __name__ == "__main__":
    main()