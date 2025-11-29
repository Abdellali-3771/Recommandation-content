"""
Popularity Recommender - Fallback pour le Cold Start
Recommande les articles les plus populaires (normalises par age)
"""

import numpy as np
import pandas as pd

class PopularityRecommender:
    """Recommandeur base sur la popularite (fallback cold start)"""
    
    def __init__(self, interactions, articles):
        """
        Parametres:
        - interactions : DataFrame avec [user_id, click_article_id, click_timestamp]
        - articles : DataFrame avec [article_id, created_at_ts, ...]
        """
        self.interactions = interactions
        self.articles = articles
        self._calculate_popularity()
    
    def _calculate_popularity(self):
        """Calcule les scores de popularite normalises par age"""
        
        # Calculer les statistiques par article
        article_stats = self.interactions.groupby('click_article_id').agg({
            'user_id': ['count', 'nunique']
        }).reset_index()
        article_stats.columns = ['article_id', 'total_clicks', 'unique_users']
        
        # Merger avec les metadonnees
        df = article_stats.merge(self.articles, on='article_id', how='left')
        
        # Convertir les timestamps
        if 'created_at_ts' in df.columns:
            df['created_date'] = pd.to_datetime(df['created_at_ts'], unit='ms')
        
        # Calculer la date de reference (dernier clic)
        if 'click_timestamp' in self.interactions.columns:
            reference_date = pd.to_datetime(self.interactions['click_timestamp'].max())
        else:
            reference_date = pd.Timestamp.now()
        
        # Calculer l'age en mois
        df['age_months'] = (reference_date - df['created_date']).dt.days / 30
        df['age_months'] = df['age_months'].clip(lower=0.1)  # Eviter division par 0
        
        # Score brut = 0.7 * unique_users + 0.3 * total_clicks
        df['raw_score'] = 0.7 * df['unique_users'] + 0.3 * df['total_clicks']
        
        # Normaliser par l'age
        df['popularity_score'] = df['raw_score'] / df['age_months']
        
        # Boost de nouveaute
        df['age_hours'] = (reference_date - df['created_date']).dt.total_seconds() / 3600
        df['novelty_boost'] = 1.0
        df.loc[df['age_hours'] < 24, 'novelty_boost'] = 1.5
        df.loc[(df['age_hours'] >= 24) & (df['age_hours'] < 72), 'novelty_boost'] = 1.2
        
        # Score final
        df['final_score'] = df['popularity_score'] * df['novelty_boost']
        
        # Trier par score
        self.popularity_scores = df[['article_id', 'final_score']].sort_values(
            'final_score', ascending=False
        )
        
        print(f"[PopularityRecommender] {len(self.popularity_scores)} articles scored")
    
    def recommend(self, user_id=None, n_recommendations=5, exclude_seen=True):
        """
        Recommande les articles les plus populaires
        
        Parametres:
        - user_id : ID utilisateur (pour exclure articles vus si exclude_seen=True)
        - n_recommendations : Nombre de recommandations
        - exclude_seen : Si True, exclure les articles deja vus
        
        Retourne:
        - Liste de tuples (article_id, score)
        """
        # Copier les scores
        scores = self.popularity_scores.copy()
        
        # Exclure les articles deja vus
        if exclude_seen and user_id is not None:
            user_articles = self.interactions[
                self.interactions['user_id'] == user_id
            ]['click_article_id'].unique()
            
            scores = scores[~scores['article_id'].isin(user_articles)]
        
        # Retourner le top N
        top_scores = scores.head(n_recommendations)
        
        return list(zip(
            top_scores['article_id'].values,
            top_scores['final_score'].values
        ))
    
    def get_popular_articles(self, n=10):
        """Retourne les N articles les plus populaires"""
        top = self.popularity_scores.head(n)
        return list(zip(top['article_id'].values, top['final_score'].values))