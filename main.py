import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN

nltk.download('wordnet')
nltk.download('stopwords')


def elbow_plot_kmeans(X):
    """
    plots the sum of squared distances to their cluster center for cluster sized between 2 and 10.
    """
    means = []
    inertia = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=100)
        km.fit(X)

        means.append(k)
        inertia.append(km.inertia_)

    plt.plot(means, inertia, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title("Kmeans elbow plot")
    plt.grid(True)
    plt.show()

def print_kmeans_cluster_keywords(vectorizer, k, cluster_center_space):
    """
    prints the keywords for each kmeans cluster
    """
    terms = vectorizer.get_feature_names_out()

    for i in range(k):
        clustering_center = cluster_center_space[i]
        top_indices = np.argsort(clustering_center)[::-1][:10]
        print(f"Cluster {i}:")
        print([terms[j] for j in top_indices])
        print("")

def plot_kmeans_clusters(X_2d, labels_kmeans):
    """
    takes a 2d mapping of the data and plots the clusters found by kmeans
    """
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_kmeans)
    plt.title("Kmeans clustering cluster visualization")
    plt.colorbar(scatter)
    plt.show()

def elbow_plot_spectral_clustering(X):
    """
    plots the silhouette score for cluster sized between 2 and 10.
    :param X:
    :return:
    """
    scores = []

    for k in range(2, 11):
        model = SpectralClustering(
            n_clusters=k,
            random_state=42
        )

        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)

        scores.append((k, score))

    ks, vals = zip(*scores)

    plt.plot(ks, vals, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("silhouette score")
    plt.title("Spectral clustering elbow plot")
    plt.grid(True)
    plt.show()

def print_spectral_cluster_keywords(vectorizer, k, X_norm, labels_spectral, svd):
    """
    prints the keywords for each spectral cluster
    """
    cluster_centers = []

    for i in range(k):
        cluster_points = X_norm[labels_spectral == i]
        center = cluster_points.mean(axis=0)
        cluster_centers.append(center)

    cluster_centers = np.array(cluster_centers)

    centers_original = svd.inverse_transform(cluster_centers)

    terms = vectorizer.get_feature_names_out()

    for i, center in enumerate(centers_original):
        top_indices = np.argsort(center)[::-1][:10]

        print(f"Spectral Cluster {i}:")
        print([terms[j] for j in top_indices])
        print("")

def plot_spectral_clusters(X_2d, labels_spectral):
    """
    takes a 2d mapping of the data and plots the clusters found by spectral clustering
    """
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_spectral)
    plt.title("Spectral Clustering cluster visualization")
    plt.colorbar(scatter)
    plt.show()


def clean_tokens(tokens):
    """
    takes a list of words and does the following:
    converts the words to lowercase
    removes non-alphanumeric characters
    removes any stop word found in the nltk's english stop word set
    removes any word shorter than three characters
    lemmatizes verbs and nouns
    removes the following words ["chastity", "skepticism", "intellect", "geb",
                                "gordon", "bank", "shameful", "surrender", "soon"]
    """
    lemmatizer = nltk.WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))

    cleaned = []

    for word in tokens:
        word = word.lower()

        if not word.isalpha():
            continue

        if word in stop_words:
            continue

        if len(word) < 3:
            continue

        word = lemmatizer.lemmatize(word, 'n')
        word = lemmatizer.lemmatize(word, 'v')

        # potential of a stopword appearing after lemmatizing, remove just in case
        if word in stop_words:
            continue

        if word in ["chastity", "skepticism", "intellect", "geb", "gordon", "bank", "shameful", "surrender", "soon"]:
            continue

        cleaned.append(word)

    return cleaned


def main():

    # DATA PREPROCESSING

    data = pd.read_csv("data/articles.csv")

    data["text"] = data["text"].apply(nltk.word_tokenize)
    data["text"] = data["text"].apply(clean_tokens)

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, min_df=2, max_df=0.80)

    tfidf_matrix = vectorizer.fit_transform(data['text'])

    svd = TruncatedSVD(n_components=100, random_state=42)
    X_reduced = svd.fit_transform(tfidf_matrix)
    X_norm = normalize(X_reduced)


    # CLUSTERING

    # elbow plots to determine a good number of clusters

    elbow_plot_kmeans(X_norm)
    elbow_plot_spectral_clustering(X_norm)

    # run clustering algorithms and assign clusters for each article

    k = 7
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=100)
    labels_kmeans = kmeans.fit_predict(X_norm)
    data['cluster_kmeans'] = labels_kmeans

    spectral_clustering = SpectralClustering(n_clusters=k, random_state=42,)
    labels_spectral = spectral_clustering.fit_predict(X_norm)
    data['cluster_spectral'] = labels_spectral

    # print the keywords for each cluster

    cluster_center_space = svd.inverse_transform(kmeans.cluster_centers_)
    print_kmeans_cluster_keywords(vectorizer, k, cluster_center_space)

    print_spectral_cluster_keywords(vectorizer, k, X_norm, labels_spectral, svd)

    # plot the clusters

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_norm)


    plot_kmeans_clusters(X_2d, labels_kmeans)
    plot_spectral_clusters(X_2d, labels_spectral)

    # calculate and print the silhouette score for algorithms

    score = silhouette_score(X_norm, labels_kmeans)
    print(f"kmeans silhouette score: {score}")

    score = silhouette_score(X_norm, labels_spectral)
    print("Spectral clustering silhouette score:", score)


    # ANOMALY DETECTION

    # distance based anomaly detection

    # calculate the distances of articles to their assigned cluster's center
    distances = kmeans.transform(X_norm)
    assigned_distances = distances[np.arange(len(X_norm)), labels_kmeans]

    # normalize the distances
    data["distance_to_center"] = assigned_distances
    data["normalized_distance"] = data.groupby("cluster_kmeans")["distance_to_center"].transform(lambda x: (x - x.mean()) / x.std())

    # mark articles with the highest normalized distance as anomalies
    outliers_kmeans = data.nlargest(170, "normalized_distance")[["doc_id", "normalized_distance", "cluster_kmeans"]]


    # DBSCAN anomaly detection

    # run the DBSCAN algorithm
    dbscan = DBSCAN(eps=0.60, min_samples=40, metric='cosine')
    labels_dbscan = dbscan.fit_predict(X_norm)
    data["cluster_dbscan"] = labels_dbscan

    # all anomalies have -1 as their label
    outliers_dbscan = data[data["cluster_dbscan"] == -1][["doc_id"]]


    # get the intersection of anomaly articles
    outliers = pd.merge(outliers_kmeans, outliers_dbscan, on="doc_id", how="inner")

    outliers = outliers.sort_values("normalized_distance", ascending=False)[:50]


    # SAVE DATA

    clusters_df = data[["doc_id", "cluster_kmeans"]].copy()
    clusters_df = clusters_df.rename(columns={"cluster_kmeans": "label"})
    clusters_df.to_csv("data/clusters.csv", index=False)

    anomalies_df = outliers[["doc_id"]].copy()
    anomalies_df = anomalies_df.reset_index(drop=True)
    anomalies_df.insert(0, "anomaly", range(1, len(anomalies_df) + 1))
    anomalies_df.to_csv("data/anomalies.csv", index=False)


    return 0

if __name__ == "__main__":
    main()

