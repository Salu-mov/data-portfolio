"""
Customer Segmentation Analysis using K-Means Clustering

Description:
    This script performs customer segmentation based on Annual Income and Spending Score.
    It generates synthetic data, determines the optimal number of clusters using the
    Elbow Method, and segments customers into distinct groups using the K-Means algorithm.
    Finally, it visualizes the clusters and prints a statistical summary for business insights.

Author: Ulas Aksac
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# --- CONFIGURATION ---
RANDOM_STATE = 42
N_SAMPLES = 200
N_CLUSTERS_FIT = 5  # Optimal k determined via Elbow Method
FIGURE_SIZE = (10, 6)


def generate_data(n_samples: int = 200) -> pd.DataFrame:
    """
    Generates a synthetic dataset representing mall customers.

    Args:
        n_samples (int): Total number of samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing 'Annual_Income_k$' and 'Spending_Score_1_100'.
    """
    np.random.seed(RANDOM_STATE)

    # Define centroids for 5 realistic customer segments
    centroids = [
        ([25, 20], [5, 5]),  # Low Income, Low Spending
        ([25, 80], [5, 5]),  # Low Income, High Spending
        ([55, 50], [10, 10]),  # Medium Income, Medium Spending
        ([85, 20], [5, 5]),  # High Income, Low Spending
        ([85, 80], [5, 5])  # High Income, High Spending
    ]

    data_groups = []
    samples_per_group = n_samples // 5

    for loc, scale in centroids:
        group = np.random.normal(loc=loc, scale=scale, size=(samples_per_group, 2))
        data_groups.append(group)

    X = np.vstack(data_groups)
    df = pd.DataFrame(X, columns=['Annual_Income_k$', 'Spending_Score_1_100'])

    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scales the data using Standard Scaler for optimal K-Means performance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled, scaler


def determine_optimal_k(X_scaled: np.ndarray, max_k: int = 10):
    """
    Plots the Elbow Method graph to help identify the optimal number of clusters.
    """
    wcss = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.grid(True)
    plt.show()


def train_model(X_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Trains the K-Means model and predicts cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters


def visualize_results(df: pd.DataFrame):
    """
    Visualizes the final customer segments using a scatter plot.
    """
    plt.figure(figsize=FIGURE_SIZE)
    sns.scatterplot(
        x='Annual_Income_k$',
        y='Spending_Score_1_100',
        hue='Cluster',
        data=df,
        palette='viridis',
        s=100,
        alpha=0.9
    )
    plt.title('Customer Segmentation Analysis (K-Means)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(title='Segment ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def print_business_insights(df: pd.DataFrame):
    """
    Prints statistical summary of clusters for business analysis.
    """
    summary = df.groupby('Cluster').mean()[['Annual_Income_k$', 'Spending_Score_1_100']]
    summary['Count'] = df['Cluster'].value_counts().sort_index()

    print("\n--- BUSINESS INSIGHTS REPORT ---")
    print(summary.round(2))
    print("-" * 40)


def main():
    # 1. Data Generation
    print("Generating synthetic data...")
    df = generate_data(n_samples=N_SAMPLES)

    # 2. Preprocessing
    X_scaled, _ = preprocess_data(df)

    # 3. Exploratory: Determine k (Uncomment to view Elbow Plot)
    # determine_optimal_k(X_scaled)

    # 4. Modeling
    print(f"Training K-Means model with k={N_CLUSTERS_FIT}...")
    clusters = train_model(X_scaled, n_clusters=N_CLUSTERS_FIT)
    df['Cluster'] = clusters

    # 5. Analysis & Visualization
    print_business_insights(df)
    visualize_results(df)


if __name__ == "__main__":
    main()