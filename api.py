"""
API FastAPI pour le systeme de recommandation
Deployable sur Google Cloud Run - AVEC LAZY LOADING
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import sys
import threading

# Ajouter src au path
sys.path.append(str(Path(__file__).parent / "src"))

from content_recommender import ContentBasedRecommender
from collaborative_recommender import CollaborativeRecommender
from popularity_recommender import PopularityRecommender

# Initialiser FastAPI
app = FastAPI(
    title="My Content Recommendation API",
    description="API de recommandation d'articles avec 3 methodes : Content-Based, Collaborative, Popularity",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour les recommandeurs
content_rec = None
collab_rec = None
popularity_rec = None
articles_df = None
interactions_df = None
models_loaded = False
loading_lock = threading.Lock()

# =====================================================
# LAZY LOADING - Charge les modeles a la 1ere requete
# =====================================================

def load_models_lazy():
    """Charge les donnees et initialise les recommandeurs (LAZY)"""
    global content_rec, collab_rec, popularity_rec, articles_df, interactions_df, models_loaded
    
    # Si deja charge, ne rien faire
    if models_loaded:
        return
    
    # Lock pour eviter les chargements multiples
    with loading_lock:
        # Double-check apres le lock
        if models_loaded:
            return
        
        print("=" * 60)
        print("🚀 CHARGEMENT LAZY DES MODELES")
        print("=" * 60)
        
        data_path = Path(__file__).parent / "data"
        
        try:
            # 1. Charger les interactions (30 fichiers pour optimiser)
            print("\n📂 Chargement des interactions...")
            clicks_dir = data_path / "clicks"
            clicks_files = sorted(list(clicks_dir.glob("clicks_hour_*.csv")))[:30]  # 30 au lieu de 50
            
            dfs = []
            for i, file in enumerate(clicks_files, 1):
                df = pd.read_csv(file)
                dfs.append(df)
                if i % 10 == 0:
                    print(f"   {i}/{len(clicks_files)} fichiers charges...")
            
            interactions_df = pd.concat(dfs, ignore_index=True)
            print(f"✅ {len(interactions_df):,} interactions chargees")
            
            # 2. Charger les metadonnees articles
            print("\n📂 Chargement des metadonnees articles...")
            articles_df = pd.read_csv(data_path / "articles_metadata.csv")
            print(f"✅ {len(articles_df):,} articles charges")
            
            # 3. Charger les embeddings PCA (100D)
            print("\n📂 Chargement des embeddings PCA...")
            embeddings_file = data_path / "articles_embeddings_pca_100D.pickle"
            
            # Fallback vers embeddings originaux si PCA non disponible
            if not embeddings_file.exists():
                print("⚠️  Fichier PCA non trouvé, utilisation des embeddings originaux...")
                embeddings_file = data_path / "articles_embeddings.pickle"
            
            with open(embeddings_file, "rb") as f:
                embeddings = pickle.load(f)
            print(f"✅ {embeddings.shape[0]:,} embeddings charges ({embeddings.shape[1]}D)")
            
            # 4. Initialiser les recommandeurs
            print("\n🔧 Initialisation des recommandeurs...")
            
            # Popularity (fallback)
            popularity_rec = PopularityRecommender(interactions_df, articles_df)
            
            # Content-Based avec embeddings PCA
            content_rec = ContentBasedRecommender(
                interactions_df,
                embeddings,
                n_components=100,
                apply_pca=False,
                min_articles_for_recs=3,
                popularity_fallback=popularity_rec
            )
            
            # Collaborative
            collab_rec = CollaborativeRecommender(
                interactions_df,
                n_components=50,
                popularity_fallback=popularity_rec
            )
            
            models_loaded = True
            
            print("\n" + "=" * 60)
            print("✅ MODELES CHARGES")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ ERREUR AU CHARGEMENT : {e}")
            import traceback
            traceback.print_exc()
            raise

def ensure_models_loaded():
    """Verifie que les modeles sont charges avant de traiter une requete"""
    if not models_loaded:
        load_models_lazy()
    
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models are still loading, please retry")

# =====================================================
# MODELES PYDANTIC
# =====================================================

class RecommendationResponse(BaseModel):
    user_id: int
    method: str
    recommendations: list
    cold_start_applied: bool = False

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    users_count: int = 0
    articles_count: int = 0
    interactions_count: int = 0

# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/", tags=["Info"])
def root():
    """Page d'accueil de l'API"""
    return {
        "message": "My Content Recommendation API",
        "status": "running",
        "models_loaded": models_loaded,
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend/{user_id}",
            "popular": "/popular",
            "users": "/users",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check():
    """Verifie que l'API fonctionne et retourne les stats"""
    
    # Si les modeles ne sont pas charges, retourner un statut partiel
    if not models_loaded:
        return {
            "status": "starting",
            "models_loaded": False,
            "users_count": 0,
            "articles_count": 0,
            "interactions_count": 0
        }
    
    return {
        "status": "healthy",
        "models_loaded": True,
        "users_count": int(interactions_df['user_id'].nunique()),
        "articles_count": len(articles_df),
        "interactions_count": len(interactions_df)
    }

@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(
    user_id: int,
    method: str = "content",
    n: int = 5
):
    """
    Genere des recommandations pour un utilisateur
    
    - **user_id**: ID de l'utilisateur
    - **method**: Methode (content, collaborative, popularity)
    - **n**: Nombre de recommandations (1-10)
    """
    
    # Charger les modeles si necessaire
    ensure_models_loaded()
    
    if not (1 <= n <= 10):
        raise HTTPException(status_code=400, detail="n must be between 1 and 10")
    
    # Determiner si cold start
    user_articles = interactions_df[interactions_df['user_id'] == user_id]['click_article_id'].nunique()
    cold_start = user_articles < 3
    
    # Generer les recommandations
    try:
        if method == "content":
            recs = content_rec.recommend(user_id, n)
        elif method == "collaborative":
            recs = collab_rec.recommend(user_id, n)
        elif method == "popularity":
            recs = popularity_rec.recommend(user_id, n)
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Use: content, collaborative, or popularity")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
    
    # Enrichir avec metadonnees
    results = []
    for article_id, score in recs:
        article_info = articles_df[articles_df['article_id'] == article_id]
        if not article_info.empty:
            results.append({
                "article_id": int(article_id),
                "score": float(score),
                "category_id": int(article_info.iloc[0]['category_id']),
                "words_count": int(article_info.iloc[0]['words_count'])
            })
    
    return {
        "user_id": user_id,
        "method": method,
        "recommendations": results,
        "cold_start_applied": cold_start
    }

@app.get("/popular", tags=["Recommendations"])
def get_popular_articles(n: int = 10):
    """
    Retourne les articles les plus populaires
    
    - **n**: Nombre d'articles (1-20)
    """
    
    # Charger les modeles si necessaire
    ensure_models_loaded()
    
    if not (1 <= n <= 20):
        raise HTTPException(status_code=400, detail="n must be between 1 and 20")
    
    try:
        popular_articles = popularity_rec.get_popular_articles(n)
        
        results = []
        for article_id, score in popular_articles:
            article_info = articles_df[articles_df['article_id'] == article_id]
            if not article_info.empty:
                results.append({
                    "article_id": int(article_id),
                    "popularity_score": float(score),
                    "category_id": int(article_info.iloc[0]['category_id']),
                    "words_count": int(article_info.iloc[0]['words_count'])
                })
        
        return {"popular_articles": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/users", tags=["Users"])
def get_users(limit: int = 20):
    """
    Retourne la liste des utilisateurs les plus actifs
    
    - **limit**: Nombre d'utilisateurs (1-50)
    """
    
    # Charger les modeles si necessaire
    ensure_models_loaded()
    
    if not (1 <= limit <= 50):
        raise HTTPException(status_code=400, detail="limit must be between 1 and 50")
    
    user_stats = interactions_df.groupby('user_id').agg({
        'click_article_id': ['count', 'nunique']
    }).reset_index()
    user_stats.columns = ['user_id', 'total_clicks', 'unique_articles']
    
    top_users = user_stats.sort_values('total_clicks', ascending=False).head(limit)
    
    return {
        "users": top_users.to_dict('records')
    }

# =====================================================
# LANCEMENT (pour test local)
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)