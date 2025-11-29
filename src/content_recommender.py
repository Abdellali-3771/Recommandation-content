"""
Content-Based Recommender avec Cold Start
Recommandation basée sur la similarité de contenu (embeddings)
Supporte les embeddings pré-réduits avec PCA
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

class ContentBasedRecommender:
    """Recommandeur basé sur la similarité de contenu avec gestion cold start"""

    def __init__(self, interactions, embeddings, n_components=None, apply_pca=False, 
                 min_articles_for_recs=3, popularity_fallback=None):
        """
        Paramètres:
        - interactions : DataFrame avec colonnes [user_id, click_article_id]
        - embeddings : Array NumPy (peut être déjà réduit avec PCA)
        - n_components : Nombre de composantes PCA (None = pas de réduction)
        - apply_pca : Si True, applique PCA sur les embeddings
        - min_articles_for_recs : Minimum d'articles lus pour faire des reco perso
        - popularity_fallback : Instance de PopularityRecommender pour cold start
        """
        self.interactions = interactions
        self.min_articles_for_recs = min_articles_for_recs
        self.popularity_fallback = popularity_fallback
        self.pca = None
        self.original_dim = embeddings.shape[1]
        
        # Appliquer PCA si demandé
        if apply_pca and n_components and n_components < embeddings.shape[1]:
            print(f"[ContentBasedRecommender] Applying PCA: {embeddings.shape[1]}D -> {n_components}D")
            self.pca = PCA(n_components=n_components, random_state=42)
            self.embeddings = self.pca.fit_transform(embeddings)
            print(f"[ContentBasedRecommender] Variance explained: {self.pca.explained_variance_ratio_.sum():.1%}")
        else:
            self.embeddings = embeddings
        
        print(f"[ContentBasedRecommender] Final embeddings shape: {self.embeddings.shape}")

    def get_embedding_info(self):
        """Retourne les infos sur les embeddings (pour Streamlit)"""
        return {
            "original_dim": self.original_dim,
            "current_dim": self.embeddings.shape[1],
            "pca_applied": self.pca is not None,
            "variance_explained_pct": self.pca.explained_variance_ratio_.sum() * 100 if self.pca else 100.0,
            "has_fallback": self.popularity_fallback is not None,
            "size_mb": self.embeddings.nbytes / (1024 * 1024)
        }

    def recommend(self, user_id, n_recommendations=5):
        """
        Recommande des articles basés sur la similarité de contenu
        Avec gestion automatique du cold start

        Retourne:
        - Liste de tuples (article_id, similarity_score)
        """
        # Récupérer les articles lus
        user_articles = self.interactions[
            self.interactions['user_id'] == user_id
        ]['click_article_id'].unique()

        # COLD START : Pas assez d'historique
        if len(user_articles) < self.min_articles_for_recs:
            print(f"[COLD START] User {user_id} has only {len(user_articles)} articles - Fallback to popularity")
            if self.popularity_fallback:
                return self.popularity_fallback.recommend(user_id, n_recommendations, exclude_seen=True)
            return []

        # Récupérer les embeddings (vérifier les limites)
        user_embeddings = []
        for article_id in user_articles:
            if 0 <= article_id < len(self.embeddings):
                user_embeddings.append(self.embeddings[article_id])

        if len(user_embeddings) == 0:
            print(f"[COLD START] No valid embeddings for user {user_id} - Fallback to popularity")
            if self.popularity_fallback:
                return self.popularity_fallback.recommend(user_id, n_recommendations, exclude_seen=True)
            return []

        # Profil utilisateur (moyenne des embeddings)
        user_profile = np.mean(user_embeddings, axis=0).reshape(1, -1)

        # Similarité avec tous les articles
        similarities = cosine_similarity(user_profile, self.embeddings)[0]

        # Exclure articles déjà lus
        all_article_ids = np.arange(len(self.embeddings))
        mask = ~np.isin(all_article_ids, user_articles)

        filtered_ids = all_article_ids[mask]
        filtered_similarities = similarities[mask]

        # Top N
        top_indices = np.argsort(filtered_similarities)[-n_recommendations:][::-1]
        top_article_ids = filtered_ids[top_indices]
        top_scores = filtered_similarities[top_indices]

        return list(zip(top_article_ids, top_scores))