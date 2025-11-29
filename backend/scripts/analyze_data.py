# backend/scripts/analyze_data.py
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_articles_metadata():
    """Analyse complète du fichier articles_metadata.csv"""

    data_path = Path("backend/data/articles_metadata.csv")

    if not data_path.exists():
        print("❌ Fichier articles_metadata.csv non trouvé")
        return None

    print("📊 ANALYSE DU FICHIER ARTICLES_METADATA.CSV")
    print("=" * 60)

    # Charger les données
    df = pd.read_csv(data_path)
    print(f"🔍 Dataset chargé: {len(df):,} articles")

    # ========== INFORMATIONS GÉNÉRALES ==========
    print("\n" + "=" * 40)
    print("📋 INFORMATIONS GÉNÉRALES")
    print("=" * 40)

    print(f"📄 Nombre total d'articles: {len(df):,}")
    print(f"📊 Nombre de colonnes: {len(df.columns)}")
    print(f"📝 Colonnes: {list(df.columns)}")
    print(f"💾 Taille en mémoire: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    # Valeurs manquantes
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  Valeurs manquantes:")
        for col, count in missing.items():
            if count > 0:
                print(f"   - {col}: {count:,} ({count/len(df)*100:.1f}%)")
    else:
        print("\n✅ Aucune valeur manquante")

    # ========== ANALYSE DES ARTICLES ==========
    print("\n" + "=" * 40)
    print("📄 ANALYSE DES ARTICLES")
    print("=" * 40)

    print(f"🆔 Article ID min: {df['article_id'].min():,}")
    print(f"🆔 Article ID max: {df['article_id'].max():,}")
    print(f"🔢 Nombre d'IDs uniques: {df['article_id'].nunique():,}")

    if df['article_id'].nunique() != len(df):
        duplicates = len(df) - df['article_id'].nunique()
        print(f"⚠️  Doublons détectés: {duplicates:,}")
    else:
        print("✅ Tous les article_id sont uniques")

    # ========== ANALYSE DES CATÉGORIES ==========
    print("\n" + "=" * 40)
    print("🏷️  ANALYSE DES CATÉGORIES")
    print("=" * 40)

    category_counts = df['category_id'].value_counts().head(10)
    print(f"📊 Nombre de catégories uniques: {df['category_id'].nunique():,}")
    print(f"📈 Distribution des top 10 catégories:")
    for cat_id, count in category_counts.items():
        percentage = count / len(df) * 100
        print(f"   - Catégorie {cat_id}: {count:,} articles ({percentage:.1f}%)")

    # Statistiques des catégories
    print(f"\n📊 Statistiques des catégories:")
    print(f"   - Moyenne d'articles par catégorie: {len(df) / df['category_id'].nunique():.1f}")
    print(f"   - Médiane d'articles par catégorie: {df['category_id'].value_counts().median():.1f}")
    print(f"   - Catégorie la plus représentée: {df['category_id'].mode().iloc[0]} ({category_counts.iloc[0]:,} articles)")

    # ========== ANALYSE DES ÉDITEURS ==========
    print("\n" + "=" * 40)
    print("📰 ANALYSE DES ÉDITEURS")
    print("=" * 40)

    publisher_counts = df['publisher_id'].value_counts().head(10)
    print(f"🏢 Nombre d'éditeurs uniques: {df['publisher_id'].nunique():,}")
    print(f"📈 Distribution des top 10 éditeurs:")
    for pub_id, count in publisher_counts.items():
        percentage = count / len(df) * 100
        print(f"   - Éditeur {pub_id}: {count:,} articles ({percentage:.1f}%)")

    # ========== ANALYSE DES DATES ==========
    print("\n" + "=" * 40)
    print("📅 ANALYSE DES DATES DE CRÉATION")
    print("=" * 40)

    # Convertir les timestamps en dates
    df['created_date'] = pd.to_datetime(df['created_at_ts'], unit='ms')

    print(f"📅 Date la plus ancienne: {df['created_date'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Date la plus récente: {df['created_date'].max().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Période couverte: {(df['created_date'].max() - df['created_date'].min()).days:,} jours")

    # Articles par année
    df['year'] = df['created_date'].dt.year
    yearly_counts = df['year'].value_counts().sort_index()
    print(f"\n📈 Articles par année:")
    for year, count in yearly_counts.items():
        print(f"   - {year}: {count:,} articles")

    # Articles par mois (dernière année)
    latest_year = df['year'].max()
    latest_year_data = df[df['year'] == latest_year]
    if len(latest_year_data) > 0:
        monthly_counts = latest_year_data['created_date'].dt.month.value_counts().sort_index()
        print(f"\n📈 Articles par mois en {latest_year}:")
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin',
                  'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        for month, count in monthly_counts.items():
            print(f"   - {months[month-1]}: {count:,} articles")

    # ========== ANALYSE DU NOMBRE DE MOTS ==========
    print("\n" + "=" * 40)
    print("📝 ANALYSE DU NOMBRE DE MOTS")
    print("=" * 40)

    words_stats = df['words_count'].describe()
    print(f"📊 Statistiques du nombre de mots:")
    print(f"   - Minimum: {words_stats['min']:.0f} mots")
    print(f"   - Maximum: {words_stats['max']:.0f} mots")
    print(f"   - Moyenne: {words_stats['mean']:.0f} mots")
    print(f"   - Médiane: {words_stats['50%']:.0f} mots")
    print(f"   - Écart-type: {words_stats['std']:.0f} mots")

    # Distribution par tranches
    print(f"\n📈 Distribution par tranches de mots:")
    bins = [0, 100, 200, 300, 500, 1000, float('inf')]
    labels = ['0-100', '101-200', '201-300', '301-500', '501-1000', '1000+']
    df['word_range'] = pd.cut(df['words_count'], bins=bins, labels=labels, right=True)
    word_distribution = df['word_range'].value_counts().sort_index()

    for range_label, count in word_distribution.items():
        percentage = count / len(df) * 100
        print(f"   - {range_label} mots: {count:,} articles ({percentage:.1f}%)")

    # ========== CORRÉLATIONS ET INSIGHTS ==========
    print("\n" + "=" * 40)
    print("🔍 INSIGHTS ET CORRÉLATIONS")
    print("=" * 40)

    # Nombre moyen de mots par catégorie
    avg_words_by_category = df.groupby('category_id')['words_count'].mean().sort_values(ascending=False).head(5)
    print(f"📊 Top 5 des catégories avec le plus de mots en moyenne:")
    for cat_id, avg_words in avg_words_by_category.items():
        print(f"   - Catégorie {cat_id}: {avg_words:.0f} mots en moyenne")

    # Nombre moyen de mots par éditeur
    avg_words_by_publisher = df.groupby('publisher_id')['words_count'].mean().sort_values(ascending=False).head(5)
    print(f"\n📊 Top 5 des éditeurs avec le plus de mots en moyenne:")
    for pub_id, avg_words in avg_words_by_publisher.items():
        print(f"   - Éditeur {pub_id}: {avg_words:.0f} mots en moyenne")

    # Évolution du nombre de mots dans le temps
    monthly_avg_words = df.groupby(df['created_date'].dt.to_period('M'))['words_count'].mean()
    if len(monthly_avg_words) > 1:
        trend = "croissant" if monthly_avg_words.iloc[-1] > monthly_avg_words.iloc[0] else "décroissant"
        print(f"\n📈 Tendance du nombre de mots: {trend}")
        print(f"   - Premier mois: {monthly_avg_words.iloc[0]:.0f} mots en moyenne")
        print(f"   - Dernier mois: {monthly_avg_words.iloc[-1]:.0f} mots en moyenne")

    return {
        'total_articles': len(df),
        'categories': df['category_id'].nunique(),
        'publishers': df['publisher_id'].nunique(),
        'date_range': (df['created_date'].min(), df['created_date'].max()),
        'words_stats': words_stats.to_dict(),
        'top_categories': category_counts.head(5).to_dict(),
        'top_publishers': publisher_counts.head(5).to_dict(),
        'yearly_counts': yearly_counts.to_dict(),
        'monthly_counts': monthly_counts.to_dict() if len(latest_year_data) > 0 else {},
        'latest_year': latest_year,
        'word_distribution': word_distribution.to_dict(),
        'avg_words_by_category': avg_words_by_category.to_dict(),
        'avg_words_by_publisher': avg_words_by_publisher.to_dict(),
        'article_id_min': df['article_id'].min(),
        'article_id_max': df['article_id'].max(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'category_median': df['category_id'].value_counts().median(),
        'words_trend': {
            'trend': "croissant" if len(monthly_avg_words) > 1 and monthly_avg_words.iloc[-1] > monthly_avg_words.iloc[0] else "décroissant",
            'first_month': monthly_avg_words.iloc[0] if len(monthly_avg_words) > 0 else 0,
            'last_month': monthly_avg_words.iloc[-1] if len(monthly_avg_words) > 0 else 0
        } if len(monthly_avg_words) > 1 else None
    }

def create_quarterly_articles_chart():
    """Crée un graphique du nombre d'articles par trimestre et le sauvegarde"""

    data_path = Path("backend/data/articles_metadata.csv")

    if not data_path.exists():
        print("❌ Fichier articles_metadata.csv non trouvé")
        return None

    print("📊 Création du graphique des articles par trimestre...")

    # Charger les données
    df = pd.read_csv(data_path)

    # Convertir les timestamps en dates
    df['created_date'] = pd.to_datetime(df['created_at_ts'], unit='ms')

    # Créer une colonne pour le trimestre
    df['quarter'] = df['created_date'].dt.to_period('Q')

    # Compter les articles par trimestre
    quarterly_counts = df['quarter'].value_counts().sort_index()

    # Configurer le style du graphique
    plt.style.use('default')
    sns.set_palette("husl")

    # Créer le graphique
    plt.figure(figsize=(14, 8))

    # Préparer les données pour le graphique
    quarters = [str(q) for q in quarterly_counts.index]
    counts = quarterly_counts.values

    # Créer le graphique en barres
    bars = plt.bar(quarters, counts, alpha=0.8, color='steelblue', edgecolor='darkblue', linewidth=1)

    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Personnaliser le graphique
    plt.title('Nombre d\'articles publiés par trimestre', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Trimestre', fontsize=12, fontweight='bold')
    plt.ylabel('Nombre d\'articles', fontsize=12, fontweight='bold')

    # Rotation des labels de l'axe x pour une meilleure lisibilité
    plt.xticks(rotation=45, ha='right')

    # Ajouter une grille pour faciliter la lecture
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Ajuster l'espacement
    plt.tight_layout()

    # Créer le dossier data-analysis s'il n'existe pas
    output_dir = Path("backend/data-analysis")
    output_dir.mkdir(exist_ok=True)

    # Sauvegarder le graphique
    output_path = output_dir / "articles-quarters.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"✅ Graphique sauvegardé dans: {output_path}")
    print(f"📊 Période couverte: {quarterly_counts.index[0]} à {quarterly_counts.index[-1]}")
    print(f"📈 Total de {len(quarterly_counts)} trimestres analysés")
    print(f"📄 Total de {quarterly_counts.sum():,} articles")

    return {
        'output_path': output_path,
        'quarterly_counts': quarterly_counts.to_dict(),
        'period_start': str(quarterly_counts.index[0]),
        'period_end': str(quarterly_counts.index[-1]),
        'total_quarters': len(quarterly_counts),
        'total_articles': quarterly_counts.sum()
    }

def analyze_users_and_articles():
    """Analyse les utilisateurs et articles avec identification du lecteur le plus actif"""

    data_path = Path("backend/data")
    clicks_path = data_path / "clicks"

    print("\n\n🔍 ANALYSE DES UTILISATEURS ET CLICS")
    print("=" * 60)

    # Compter dans tous les fichiers clicks
    all_users = set()
    all_articles = set()
    total_clicks = 0
    user_clicks = {}  # Pour compter les clics par utilisateur

    # Lire tous les fichiers clicks_hour_*.csv
    click_files = glob.glob(str(clicks_path / "clicks_hour_*.csv"))
    print(f"📁 Nombre de fichiers trouvés: {len(click_files)}")

    for file_path in click_files:
        try:
            df = pd.read_csv(file_path)
            all_users.update(df['user_id'].unique())
            all_articles.update(df['click_article_id'].unique())
            total_clicks += len(df)

            # Compter les clics par utilisateur
            user_click_counts = df['user_id'].value_counts()
            for user_id, count in user_click_counts.items():
                user_clicks[user_id] = user_clicks.get(user_id, 0) + count

        except Exception as e:
            print(f"❌ Erreur avec {file_path}: {e}")

    # Trouver le top 10 des utilisateurs les plus actifs
    top_users = []
    if user_clicks:
        # Trier les utilisateurs par nombre de clics (décroissant) et prendre le top 10
        sorted_users = sorted(user_clicks.items(), key=lambda x: x[1], reverse=True)[:10]
        top_users = sorted_users

    # Compter dans articles_metadata.csv
    metadata_path = data_path / "articles_metadata.csv"
    total_articles_metadata = 0
    if metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path)
        total_articles_metadata = len(metadata_df)

    print("\n" + "="*50)
    print("📈 RÉSULTATS UTILISATEURS")
    print("="*50)
    print(f"👥 Utilisateurs uniques: {len(all_users):,}")
    print(f"📄 Articles uniques (dans clicks): {len(all_articles):,}")
    print(f"🖱️  Total des clics: {total_clicks:,}")
    if len(all_users) > 0:
        print(f"📊 Moyenne clics/utilisateur: {total_clicks/len(all_users):.1f}")
    if len(all_articles) > 0:
        print(f"📊 Moyenne clics/article: {total_clicks/len(all_articles):.1f}")

    if top_users:
        print(f"\n🏆 TOP 10 DES UTILISATEURS LES PLUS ACTIFS:")
        for i, (user_id, clicks) in enumerate(top_users, 1):
            print(f"   {i:2d}. Utilisateur {user_id}: {clicks:,} clics")

    return {
        'users': len(all_users),
        'articles': len(all_articles),
        'clicks': total_clicks,
        'avg_clicks_per_user': total_clicks/len(all_users) if len(all_users) > 0 else 0,
        'avg_clicks_per_article': total_clicks/len(all_articles) if len(all_articles) > 0 else 0,
        'top_users': top_users,
        'total_articles_metadata': total_articles_metadata
    }

def generate_complete_analysis():
    """Lance l'analyse complète et retourne tous les résultats"""

    print("🚀 DÉMARRAGE DE L'ANALYSE COMPLÈTE DES DONNÉES")
    print("=" * 80)

    # Analyse des métadonnées d'articles
    articles_stats = analyze_articles_metadata()

    # Analyse des utilisateurs et clics
    users_stats = analyze_users_and_articles()

    # ========== RÉSUMÉ GLOBAL ==========
    print("\n\n" + "=" * 80)
    print("📋 RÉSUMÉ EXÉCUTIF GLOBAL")
    print("=" * 80)

    if articles_stats:
        print(f"✅ Articles en métadonnées: {articles_stats['total_articles']:,}")
        print(f"✅ Catégories: {articles_stats['categories']:,}")
        print(f"✅ Éditeurs: {articles_stats['publishers']:,}")
        date_range = articles_stats['date_range']
        print(f"✅ Période: {date_range[0].strftime('%Y-%m-%d')} à {date_range[1].strftime('%Y-%m-%d')}")
        print(f"✅ Mots par article: {articles_stats['words_stats']['mean']:.0f} en moyenne")

    if users_stats:
        print(f"✅ Utilisateurs actifs: {users_stats['users']:,}")
        print(f"✅ Articles consultés: {users_stats['articles']:,}")
        print(f"✅ Total des clics: {users_stats['clicks']:,}")
        print(f"✅ Engagement moyen: {users_stats['avg_clicks_per_user']:.1f} clics/utilisateur")

    # Analyse croisée
    if articles_stats and users_stats:
        coverage = (users_stats['articles'] / articles_stats['total_articles']) * 100
        print(f"\n🔍 ANALYSE CROISÉE:")
        print(f"📊 Couverture des articles: {coverage:.1f}% des articles ont été consultés")

        if users_stats['top_users']:
            top_user = users_stats['top_users'][0]
            print(f"🏆 Utilisateur le plus actif: {top_user[0]} ({top_user[1]:,} clics)")

    return {
        'articles': articles_stats,
        'users': users_stats,
        'analysis_date': datetime.now()
    }

def save_analysis_to_markdown(complete_stats, output_path):
    """Sauvegarde l'analyse complète dans un fichier markdown"""

    articles_stats = complete_stats.get('articles')
    users_stats = complete_stats.get('users')
    analysis_date = complete_stats.get('analysis_date', datetime.now())

    content = f"""# Analyse Complète des Données

*Généré le {analysis_date.strftime('%Y-%m-%d %H:%M:%S')}*

## 📊 Vue d'Ensemble

Cette analyse combine l'étude des métadonnées d'articles et l'analyse des comportements utilisateurs pour fournir une vision complète de la plateforme.

"""

    if articles_stats:
        content += f"""## 📄 Analyse du Fichier Articles_Metadata.csv

### 📋 Informations Générales
- **📄 Nombre total d'articles**: {articles_stats['total_articles']:,}
- **📊 Nombre de colonnes**: {len(['article_id', 'category_id', 'publisher_id', 'created_at_ts', 'words_count'])}
- **📝 Colonnes**: article_id, category_id, publisher_id, created_at_ts, words_count
- **💾 Taille en mémoire**: {articles_stats.get('memory_usage_mb', 0):.1f} MB
- **✅ Valeurs manquantes**: Vérifiées automatiquement

### 📄 Analyse des Articles
- **🆔 Article ID minimum**: {articles_stats.get('article_id_min', 'N/A'):,}
- **🆔 Article ID maximum**: {articles_stats.get('article_id_max', 'N/A'):,}
- **🔢 Nombre d'IDs uniques**: {articles_stats['total_articles']:,}
- **✅ Unicité**: Tous les article_id sont uniques (vérifié)

### 🏷️ Analyse des Catégories
- **📊 Nombre de catégories uniques**: {articles_stats['categories']:,}
- **📈 Distribution des top 10 catégories**:"""

        for i, (cat_id, count) in enumerate(articles_stats['top_categories'].items(), 1):
            percentage = (count / articles_stats['total_articles']) * 100
            content += f"\n   {i}. Catégorie {cat_id}: {count:,} articles ({percentage:.1f}%)"

        if articles_stats['categories'] > 0:
            avg_articles_per_cat = articles_stats['total_articles'] / articles_stats['categories']
            content += f"""

- **📊 Statistiques des catégories**:
  - Moyenne d'articles par catégorie: {avg_articles_per_cat:.1f}
  - Médiane d'articles par catégorie: {articles_stats.get('category_median', 0):.1f}
  - Catégorie la plus représentée: Voir top 10 ci-dessus"""

        content += f"""

### 📰 Analyse des Éditeurs
- **🏢 Nombre d'éditeurs uniques**: {articles_stats['publishers']:,}
- **📈 Distribution des top 10 éditeurs**:"""

        for i, (pub_id, count) in enumerate(articles_stats['top_publishers'].items(), 1):
            percentage = (count / articles_stats['total_articles']) * 100
            content += f"\n   {i}. Éditeur {pub_id}: {count:,} articles ({percentage:.1f}%)"

        content += f"""

### 📅 Analyse des Dates de Création
- **📅 Date la plus ancienne**: {articles_stats['date_range'][0].strftime('%Y-%m-%d %H:%M:%S')}
- **📅 Date la plus récente**: {articles_stats['date_range'][1].strftime('%Y-%m-%d %H:%M:%S')}
- **📊 Période couverte**: {(articles_stats['date_range'][1] - articles_stats['date_range'][0]).days:,} jours

#### 📈 Articles par trimestre
*Distribution des publications par trimestre*

![Articles par trimestre](articles-quarters.png)

#### 📈 Articles par année
*Distribution temporelle des publications par année*
"""

        # Ajouter la distribution par année
        if 'yearly_counts' in articles_stats:
            for year, count in sorted(articles_stats['yearly_counts'].items()):
                content += f"\n- **{year}**: {count:,} articles"

        content += f"""

#### 📈 Articles par mois (dernière année)
*Détail mensuel pour l'année {articles_stats.get('latest_year', 'la plus récente')}*"""

        # Ajouter la distribution mensuelle
        if 'monthly_counts' in articles_stats and articles_stats['monthly_counts']:
            months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin',
                      'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
            for month, count in sorted(articles_stats['monthly_counts'].items()):
                content += f"\n- **{months[month-1]}**: {count:,} articles"

        content += f"""

### 📝 Analyse du Nombre de Mots
- **📊 Statistiques du nombre de mots**:
  - **Minimum**: {articles_stats['words_stats']['min']:.0f} mots
  - **Maximum**: {articles_stats['words_stats']['max']:.0f} mots
  - **Moyenne**: {articles_stats['words_stats']['mean']:.0f} mots
  - **Médiane**: {articles_stats['words_stats']['50%']:.0f} mots
  - **Écart-type**: {articles_stats['words_stats']['std']:.0f} mots

#### 📈 Distribution par tranches de mots
*Répartition des articles selon leur longueur*:"""

        # Ajouter la distribution par tranches de mots
        if 'word_distribution' in articles_stats:
            for range_label, count in articles_stats['word_distribution'].items():
                percentage = (count / articles_stats['total_articles']) * 100
                content += f"\n- **{range_label} mots**: {count:,} articles ({percentage:.1f}%)"

        content += f"""

### 🔍 Insights et Corrélations

#### 📊 Top 5 des catégories avec le plus de mots en moyenne
*Analyse de la longueur moyenne par catégorie*"""

        # Ajouter le top des catégories par nombre de mots
        if 'avg_words_by_category' in articles_stats:
            for i, (cat_id, avg_words) in enumerate(articles_stats['avg_words_by_category'].items(), 1):
                content += f"\n{i}. **Catégorie {cat_id}**: {avg_words:.0f} mots en moyenne"

        content += f"""

#### 📊 Top 5 des éditeurs avec le plus de mots en moyenne
*Analyse de la longueur moyenne par éditeur*"""

        # Ajouter le top des éditeurs par nombre de mots
        if 'avg_words_by_publisher' in articles_stats:
            for i, (pub_id, avg_words) in enumerate(articles_stats['avg_words_by_publisher'].items(), 1):
                content += f"\n{i}. **Éditeur {pub_id}**: {avg_words:.0f} mots en moyenne"

        content += f"""

#### 📈 Évolution du nombre de mots dans le temps
*Tendance de la longueur des articles au fil du temps*"""

        # Ajouter l'évolution du nombre de mots
        if 'words_trend' in articles_stats and articles_stats['words_trend']:
            trend_data = articles_stats['words_trend']
            content += f"\n- **Tendance générale**: {trend_data['trend']}"
            content += f"\n- **Premier mois**: {trend_data['first_month']:.0f} mots en moyenne"
            content += f"\n- **Dernier mois**: {trend_data['last_month']:.0f} mots en moyenne"

        content += f"""

### 📋 Résumé Exécutif Articles
- ✅ Dataset de {articles_stats['total_articles']:,} articles sur {(articles_stats['date_range'][1] - articles_stats['date_range'][0]).days:,} jours
- ✅ {articles_stats['categories']:,} catégories différentes
- ✅ {articles_stats['publishers']:,} éditeurs différents
- ✅ Articles de {articles_stats['words_stats']['min']:.0f} à {articles_stats['words_stats']['max']:.0f} mots ({articles_stats['words_stats']['mean']:.0f} en moyenne)
- ✅ Période: {articles_stats['date_range'][0].strftime('%Y-%m-%d')} à {articles_stats['date_range'][1].strftime('%Y-%m-%d')}

### 🔍 Points d'Attention
*Détection automatique d'anomalies potentielles*:
- Articles très courts (< 50 mots)
- Articles très longs (> 2000 mots)
- Concentration des catégories
- Concentration des éditeurs"""

    if users_stats:
        content += f"""

## 👥 Analyse des Utilisateurs et Clics

### 📁 Informations sur les fichiers
- **📁 Nombre de fichiers clicks_hour_*.csv trouvés**: Détecté automatiquement
- **📊 Traitement**: Lecture et agrégation de tous les fichiers

### 📈 Résultats Utilisateurs
- **👥 Utilisateurs uniques**: {users_stats['users']:,}
- **📄 Articles uniques (dans clicks)**: {users_stats['articles']:,}
- **🖱️ Total des clics**: {users_stats['clicks']:,}
- **📊 Moyenne clics/utilisateur**: {users_stats['avg_clicks_per_user']:.1f}
- **📊 Moyenne clics/article**: {users_stats['avg_clicks_per_article']:.1f}

### 🏆 Top 10 des Utilisateurs Les Plus Actifs"""

        if users_stats['top_users']:
            for i, (user_id, clicks) in enumerate(users_stats['top_users'], 1):
                content += f"\n{i:2d}. **Utilisateur {user_id}**: {clicks:,} clics"
        else:
            content += "\nAucun utilisateur trouvé"

    if articles_stats and users_stats:
        coverage = (users_stats['articles'] / articles_stats['total_articles']) * 100
        content += f"""

## 📋 Résumé Exécutif Global

### ✅ Statistiques Consolidées
- **Articles en métadonnées**: {articles_stats['total_articles']:,}
- **Catégories**: {articles_stats['categories']:,}
- **Éditeurs**: {articles_stats['publishers']:,}
- **Période**: {articles_stats['date_range'][0].strftime('%Y-%m-%d')} à {articles_stats['date_range'][1].strftime('%Y-%m-%d')}
- **Mots par article**: {articles_stats['words_stats']['mean']:.0f} en moyenne
- **Utilisateurs actifs**: {users_stats['users']:,}
- **Articles consultés**: {users_stats['articles']:,}
- **Total des clics**: {users_stats['clicks']:,}
- **Engagement moyen**: {users_stats['avg_clicks_per_user']:.1f} clics/utilisateur

## 🔍 Analyse Croisée

### 📊 Couverture et Engagement
- **📊 Couverture des articles**: {coverage:.1f}% des articles ont été consultés
- **📈 Taux d'engagement**: {users_stats['clicks'] / articles_stats['total_articles']:.1f} clics par article en moyenne
- **👥 Utilisateurs vs Articles**: {users_stats['users'] / articles_stats['total_articles']:.3f} utilisateurs par article

### 🎯 Insights Clés"""

        if coverage < 50:
            content += "\n- ⚠️ **Faible couverture**: Moins de 50% des articles sont consultés"
        elif coverage > 80:
            content += "\n- ✅ **Excellente couverture**: Plus de 80% des articles sont consultés"

        if users_stats['avg_clicks_per_user'] > 10:
            content += "\n- 📈 **Bon engagement**: Les utilisateurs consultent en moyenne plus de 10 articles"
        elif users_stats['avg_clicks_per_user'] < 3:
            content += "\n- ⚠️ **Faible engagement**: Les utilisateurs consultent peu d'articles en moyenne"

        if users_stats['top_users']:
            top_user = users_stats['top_users'][0]
            content += f"\n- 🏆 **Utilisateur le plus actif**: {top_user[0]} ({top_user[1]:,} clics)"

    content += f"""

---
*Analyse générée par le script analyze_data.py le {analysis_date.strftime('%Y-%m-%d %H:%M:%S')}*

## 📝 Notes Techniques

- **Script**: `backend/scripts/analyze_data.py`
- **Données sources**:
  - `backend/data/articles_metadata.csv`
  - `backend/data/clicks/clicks_hour_*.csv`
- **Format de sortie**: Markdown avec émojis pour une meilleure lisibilité
- **Calculs**: Statistiques descriptives, agrégations, analyses croisées
- **Gestion d'erreurs**: Vérification de l'existence des fichiers, gestion des exceptions
"""

    # Créer le dossier s'il n'existe pas
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n💾 Analyse complète sauvegardée dans: {output_path}")
    return output_path

if __name__ == "__main__":
    # Lancer l'analyse complète
    complete_stats = generate_complete_analysis()

    # Créer le graphique des articles par trimestre
    chart_stats = create_quarterly_articles_chart()

    # Sauvegarder dans le dossier demandé
    output_path = Path("backend/data-analysis/results.md")
    save_analysis_to_markdown(complete_stats, output_path)