"""
Collaborative Filtering Recommender avec Cold Start
Recommandation basee sur le filtrage collaboratif (SVD)
Avec fallback automatique vers popularite pour nouveaux utilisateurs
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    """Recommandeur base sur le filtrage collaboratif avec cold start"""
    
    def __init__(self, interactions, n_components=50, popularity_fallback=None):
        """
        Parametres:
        - interactions : DataFrame avec [user_id, click_article_id]
        - n_components : Nombre de dimensions latentes pour SVD
        - popularity_fallback : Instance de PopularityRecommender pour fallback
        """
        self.interactions = interactions
        self.n_components = n_components
        self.popularity_fallback = popularity_fallback
        self._build_model()
    
    def _build_model(self):
        """Construit le modele SVD"""
        # Calculer les ratings (nb de clics)
        rating_data = self.interactions.groupby(['user_id', 'click_article_id']).size().reset_index(name='rating')
        
        # Creer les mappings
        unique_users = rating_data['user_id'].unique()
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        
        unique_articles = rating_data['click_article_id'].unique()
        self.article_to_idx = {aid: idx for idx, aid in enumerate(unique_articles)}
        self.idx_to_article = {idx: aid for aid, idx in self.article_to_idx.items()}
        
        # Creer la matrice user-item
        user_indices = rating_data['user_id'].map(self.user_to_idx).values
        article_indices = rating_data['click_article_id'].map(self.article_to_idx).values
        ratings = rating_data['rating'].values
        
        self.user_item_matrix = csr_matrix(
            (ratings, (user_indices, article_indices)),
            shape=(len(unique_users), len(unique_articles))
        )
        
        # Appliquer SVD
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_features = svd.fit_transform(self.user_item_matrix)
        self.article_features = svd.components_.T
        
        print(f"[CollaborativeRecommender] {len(unique_users)} users, {len(unique_articles)} articles")
    
    def recommend(self, user_id, n_recommendations=5):
        """
        Recommande des articles avec gestion du cold start
        
        Retourne:
        - Liste de tuples (article_id, score)
        - OU fallback vers popularite si utilisateur inconnu
        """
        # COLD START : Utilisateur inconnu
        if user_id not in self.user_to_idx:
            print(f"[COLD START] User {user_id} not in training data - Fallback to popularity")
            if self.popularity_fallback:
                return self.popularity_fallback.recommend(user_id, n_recommendations, exclude_seen=False)
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_features[user_idx].reshape(1, -1)
        
        # Scores de similarite
        scores = cosine_similarity(user_vector, self.article_features)[0]
        
        # Exclure articles vus
        user_seen = self.user_item_matrix[user_idx].nonzero()[1]
        scores[user_seen] = -1
        
        # Top N
        top_indices = np.argsort(scores)[-n_recommendations:][::-1]
        top_article_ids = [self.idx_to_article[idx] for idx in top_indices]
        top_scores = scores[top_indices]
        
        return list(zip(top_article_ids, top_scores))