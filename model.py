import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os


# --- Smarter genre rules based on feature ranges ---
def rule_based_genre(row):
    """
    Assigns a genre label based on audio feature thresholds.
    This makes classification more musically meaningful than raw cluster IDs.
    """
    tempo       = row.get('tempo', 120)
    energy      = row.get('energy', 0.5)
    danceability = row.get('danceability', 0.5)
    valence     = row.get('valence', 0.5)      # optional: positivity of track
    acousticness = row.get('acousticness', 0.5) # optional

    # High energy + high tempo + high dance → Dance / EDM
    if energy > 0.75 and tempo > 120 and danceability > 0.7:
        return 'Dance / EDM'

    # High energy + high tempo + low dance → Rock / Metal
    if energy > 0.75 and tempo > 120 and danceability <= 0.7:
        return 'Rock / Metal'

    # Moderate energy + high dance + moderate tempo → Pop
    if 0.4 < energy <= 0.75 and danceability > 0.6 and 90 <= tempo <= 140:
        return 'Pop'

    # Low energy + low tempo + high acousticness → Chill / Acoustic
    if energy <= 0.4 and tempo < 100:
        return 'Chill / Acoustic'

    # Low energy + high valence → Soft / Indie
    if energy <= 0.5 and valence > 0.5:
        return 'Soft / Indie'

    # Very high energy catch-all
    if energy > 0.85:
        return 'High Energy / Electronic'

    return 'Other / Ambient'


def find_optimal_clusters(scaled_features, max_k=8):
    """
    Uses silhouette score to find the optimal number of clusters
    instead of always hardcoding k=5.
    """
    n_samples = len(scaled_features)
    max_k = min(max_k, n_samples - 1)  # Can't have more clusters than samples

    if max_k < 2:
        return 2  # Minimum meaningful clusters

    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def process_music_data(filepath):
    df = pd.read_csv(filepath)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()

    # Core features always required
    core_features = ['tempo', 'energy', 'danceability']

    # Optional enrichment features — used if present
    optional_features = ['valence', 'acousticness', 'instrumentalness', 'loudness', 'speechiness']
    available_optional = [f for f in optional_features if f in df.columns]

    all_features = core_features + available_optional

    # Drop rows with NaN in used columns
    df = df.dropna(subset=core_features)

    features = df[all_features].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # --- Smart cluster count selection ---
    optimal_k = find_optimal_clusters(scaled_features)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    df['Cluster'] = clusters

    # --- Rule-based genre assignment (smarter than fixed cluster→genre map) ---
    df['Predicted_Genre'] = df.apply(rule_based_genre, axis=1)

    # --- Build summary stats ---
    genre_counts = df['Predicted_Genre'].value_counts().to_dict()
    cluster_counts = df['Cluster'].value_counts().sort_index().to_dict()

    stats = {
        'total_songs': len(df),
        'num_clusters': optimal_k,
        'genre_distribution': genre_counts,
        'avg_tempo': round(df['tempo'].mean(), 1),
        'avg_energy': round(df['energy'].mean(), 3),
        'avg_danceability': round(df['danceability'].mean(), 3),
    }

    # --- Save processed CSV ---
    os.makedirs('processed', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    output_path = os.path.join('processed', 'grouped_music.csv')
    df.to_csv(output_path, index=False)

    # --- Plot: Tempo vs Energy coloured by cluster ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f0f')

    colors = cm.plasma(np.linspace(0.1, 0.9, optimal_k))

    # Scatter plot
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a2e')
    for c in range(optimal_k):
        mask = clusters == c
        ax1.scatter(
            scaled_features[mask, 0],
            scaled_features[mask, 1],
            color=colors[c],
            alpha=0.7,
            s=50,
            label=f'Cluster {c}'
        )
    ax1.set_xlabel('Tempo (scaled)', color='#ccc')
    ax1.set_ylabel('Energy (scaled)', color='#ccc')
    ax1.set_title('Music Clusters: Tempo vs Energy', color='white', fontsize=13)
    ax1.tick_params(colors='#aaa')
    ax1.spines[:].set_color('#333')
    ax1.legend(facecolor='#111', labelcolor='white', fontsize=8)

    # Genre bar chart
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())
    bar_colors = cm.plasma(np.linspace(0.1, 0.9, len(genres)))
    bars = ax2.barh(genres, counts, color=bar_colors)
    ax2.set_xlabel('Number of Songs', color='#ccc')
    ax2.set_title('Genre Distribution', color='white', fontsize=13)
    ax2.tick_params(colors='#aaa')
    ax2.spines[:].set_color('#333')
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(count), va='center', color='white', fontsize=9)

    plt.tight_layout(pad=2)
    plot_path = os.path.join('static', 'cluster_plot.png')
    plt.savefig(plot_path, facecolor='#0f0f0f', dpi=150)
    plt.close()

    return 'grouped_music.csv', 'cluster_plot.png', stats
