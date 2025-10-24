"""
Tests für das Clustering-Modul
"""

import pytest
import numpy as np
from app.cluster import Clusterer, find_optimal_k, estimate_eps


class TestClusterer:
    """Tests für Clusterer"""
    
    def test_init(self):
        """Test Initialisierung"""
        clusterer = Clusterer(random_state=42)
        assert clusterer.random_state == 42
        assert clusterer.model is None
        assert clusterer.method is None
        assert not clusterer.is_fitted
    
    def test_kmeans_clustering(self):
        """Test K-Means Clustering"""
        # Einfache Test-Daten
        X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [10, 10], [10, 11], [11, 10], [11, 11]])
        
        clusterer = Clusterer(random_state=42)
        labels = clusterer.fit_predict(X, method='kmeans', n_clusters=2)
        
        assert len(labels) == len(X)
        assert clusterer.is_fitted
        assert clusterer.method == 'kmeans'
        assert clusterer.n_clusters_ == 2
        assert clusterer.outlier_count == 0
    
    def test_dbscan_clustering(self):
        """Test DBSCAN Clustering"""
        # Test-Daten mit zwei Clustern
        X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [10, 10], [10, 11], [11, 10], [11, 11]])
        
        clusterer = Clusterer(random_state=42)
        labels = clusterer.fit_predict(X, method='dbscan', eps=2.0, min_samples=2)
        
        assert len(labels) == len(X)
        assert clusterer.is_fitted
        assert clusterer.method == 'dbscan'
        assert clusterer.n_clusters_ >= 1
    
    def test_hdbscan_clustering(self):
        """Test HDBSCAN Clustering"""
        # Test-Daten
        X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [10, 10], [10, 11], [11, 10], [11, 11]])
        
        clusterer = Clusterer(random_state=42)
        labels = clusterer.fit_predict(X, method='hdbscan', min_cluster_size=2)
        
        assert len(labels) == len(X)
        assert clusterer.is_fitted
        assert clusterer.method == 'hdbscan'
    
    def test_agglomerative_clustering(self):
        """Test Agglomerative Clustering"""
        # Test-Daten
        X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [10, 10], [10, 11], [11, 10], [11, 11]])
        
        clusterer = Clusterer(random_state=42)
        labels = clusterer.fit_predict(X, method='agglomerative', n_clusters=2)
        
        assert len(labels) == len(X)
        assert clusterer.is_fitted
        assert clusterer.method == 'agglomerative'
        assert clusterer.n_clusters_ == 2
    
    def test_get_cluster_info(self):
        """Test Cluster-Informationen"""
        X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [10, 10], [10, 11], [11, 10], [11, 11]])
        
        clusterer = Clusterer(random_state=42)
        labels = clusterer.fit_predict(X, method='kmeans', n_clusters=2)
        
        info = clusterer.get_cluster_info()
        
        assert 'method' in info
        assert 'n_clusters' in info
        assert 'outlier_count' in info
        assert 'cluster_sizes' in info
        assert info['method'] == 'kmeans'
        assert info['n_clusters'] == 2


class TestClusteringUtilities:
    """Tests für Clustering-Utilities"""
    
    def test_find_optimal_k(self):
        """Test K-Optimierung"""
        # Test-Daten mit klaren Clustern
        X = np.array([
            [1, 1], [1, 2], [2, 1], [2, 2],  # Cluster 1
            [10, 10], [10, 11], [11, 10], [11, 11],  # Cluster 2
            [20, 20], [20, 21], [21, 20], [21, 21]  # Cluster 3
        ])
        
        result = find_optimal_k(X, k_range=(2, 5), method='kmeans', random_state=42)
        
        assert 'best_k' in result
        assert 'best_score' in result
        assert 'k_values' in result
        assert 'scores' in result
        assert result['best_k'] >= 2
        assert result['best_k'] <= 5
        assert len(result['k_values']) == 4  # 2, 3, 4, 5
    
    def test_estimate_eps(self):
        """Test eps-Schätzung für DBSCAN"""
        # Test-Daten
        X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [10, 10], [10, 11], [11, 10], [11, 11]])
        
        eps = estimate_eps(X, min_samples=2, k=4)
        
        assert isinstance(eps, float)
        assert eps > 0
        assert eps < 10  # Sollte vernünftigen Wert haben
