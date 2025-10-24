"""
Clustering-Modul für KMeans, DBSCAN, HDBSCAN und Agglomerative Clustering
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None

import warnings


class Clusterer:
    """Klasse für verschiedene Clustering-Algorithmen"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.method = None
        self.labels_ = None
        self.n_clusters_ = 0
        self.outlier_count = 0
        self.is_fitted = False
        
    def fit_predict(
        self, 
        X: np.ndarray, 
        method: str = 'kmeans',
        **kwargs
    ) -> np.ndarray:
        """
        Führt Clustering durch
        
        Args:
            X: Eingabedaten
            method: Clustering-Methode ('kmeans', 'dbscan', 'hdbscan', 'agglomerative')
            **kwargs: Zusätzliche Parameter für die jeweilige Methode
        """
        self.method = method
        
        if method == 'kmeans':
            return self._fit_predict_kmeans(X, **kwargs)
        elif method == 'dbscan':
            return self._fit_predict_dbscan(X, **kwargs)
        elif method == 'hdbscan':
            return self._fit_predict_hdbscan(X, **kwargs)
        elif method == 'agglomerative':
            return self._fit_predict_agglomerative(X, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _fit_predict_kmeans(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """KMeans Clustering"""
        # Standard-Parameter
        kmeans_params = {
            'n_clusters': 8,
            'n_init': 'auto',
            'random_state': self.random_state,
            'max_iter': 300
        }
        
        # Benutzer-Parameter überschreiben
        kmeans_params.update(kwargs)
        
        self.model = KMeans(**kmeans_params)
        self.labels_ = self.model.fit_predict(X)
        self.n_clusters_ = len(np.unique(self.labels_))
        self.outlier_count = 0
        self.is_fitted = True
        
        return self.labels_
    
    def _fit_predict_dbscan(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """DBSCAN Clustering"""
        # Standard-Parameter
        dbscan_params = {
            'eps': 0.5,
            'min_samples': 5,
            'metric': 'euclidean'
        }
        
        # Benutzer-Parameter überschreiben
        dbscan_params.update(kwargs)
        
        self.model = DBSCAN(**dbscan_params)
        self.labels_ = self.model.fit_predict(X)
        
        # Noise-Punkte (Label -1) als Outlier zählen
        unique_labels = np.unique(self.labels_)
        self.n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.outlier_count = np.sum(self.labels_ == -1)
        self.is_fitted = True
        
        return self.labels_
    
    def _fit_predict_hdbscan(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """HDBSCAN Clustering"""
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN ist nicht verfügbar. Bitte installieren Sie hdbscan.")
        
        # Standard-Parameter
        hdbscan_params = {
            'min_cluster_size': 5,
            'min_samples': None,
            'metric': 'euclidean',
            'cluster_selection_epsilon': 0.0,
            'cluster_selection_method': 'eom'
        }
        
        # Benutzer-Parameter überschreiben
        hdbscan_params.update(kwargs)
        
        self.model = hdbscan.HDBSCAN(**hdbscan_params)
        self.labels_ = self.model.fit_predict(X)
        
        # Noise-Punkte (Label -1) als Outlier zählen
        unique_labels = np.unique(self.labels_)
        self.n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.outlier_count = np.sum(self.labels_ == -1)
        self.is_fitted = True
        
        return self.labels_
    
    def _fit_predict_agglomerative(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Agglomerative Clustering"""
        # Standard-Parameter
        agglo_params = {
            'n_clusters': 8,
            'linkage': 'ward',
            'metric': 'euclidean'
        }
        
        # Benutzer-Parameter überschreiben
        agglo_params.update(kwargs)
        
        self.model = AgglomerativeClustering(**agglo_params)
        self.labels_ = self.model.fit_predict(X)
        self.n_clusters_ = len(np.unique(self.labels_))
        self.outlier_count = 0
        self.is_fitted = True
        
        return self.labels_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage für neue Daten"""
        if not self.is_fitted:
            raise ValueError("Model muss zuerst gefittet werden")
        
        if self.method in ['dbscan', 'hdbscan']:
            # DBSCAN und HDBSCAN können keine neuen Daten vorhersagen
            raise ValueError(f"Predict für {self.method} nicht unterstützt")
        
        return self.model.predict(X)
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Gibt Informationen über die Cluster zurück"""
        if not self.is_fitted:
            return {}
        
        unique_labels = np.unique(self.labels_)
        cluster_sizes = {}
        
        for label in unique_labels:
            if label == -1:  # Noise/Outlier
                cluster_sizes['outlier'] = np.sum(self.labels_ == label)
            else:
                cluster_sizes[f'cluster_{label}'] = np.sum(self.labels_ == label)
        
        return {
            'method': self.method,
            'n_clusters': self.n_clusters_,
            'outlier_count': self.outlier_count,
            'cluster_sizes': cluster_sizes,
            'total_points': len(self.labels_),
            'outlier_percentage': (self.outlier_count / len(self.labels_)) * 100 if len(self.labels_) > 0 else 0
        }


def find_optimal_k(
    X: np.ndarray, 
    k_range: Tuple[int, int] = (2, 15),
    method: str = 'kmeans',
    metric: str = 'silhouette',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Findet die optimale Anzahl von Clustern
    
    Args:
        X: Eingabedaten
        k_range: Bereich für k (min, max)
        method: Clustering-Methode
        metric: Metrik für Optimierung ('silhouette', 'inertia')
        random_state: Random Seed
    """
    k_min, k_max = k_range
    k_values = range(k_min, k_max + 1)
    
    scores = []
    inertias = []
    
    for k in k_values:
        if method == 'kmeans':
            model = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
        elif method == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=k)
        else:
            raise ValueError(f"Optimal k search not supported for {method}")
        
        labels = model.fit_predict(X)
        
        # Silhouette Score
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X, labels)
            scores.append(sil_score)
        else:
            scores.append(-1)
        
        # Inertia (nur für KMeans)
        if method == 'kmeans':
            inertias.append(model.inertia_)
    
    # Beste k finden
    if metric == 'silhouette':
        best_k = k_values[np.argmax(scores)]
        best_score = max(scores)
    else:  # inertia
        best_k = k_values[np.argmin(inertias)]
        best_score = min(inertias)
    
    return {
        'best_k': best_k,
        'best_score': best_score,
        'k_values': list(k_values),
        'scores': scores,
        'inertias': inertias if method == 'kmeans' else None,
        'metric': metric
    }


def estimate_eps(X: np.ndarray, min_samples: int = 5, k: int = 4) -> float:
    """
    Schätzt eps für DBSCAN basierend auf k-distance Plot
    
    Args:
        X: Eingabedaten
        min_samples: Mindestanzahl von Samples
        k: Anzahl der Nachbarn für k-distance
    """
    from sklearn.neighbors import NearestNeighbors
    
    # k-nearest neighbors finden
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # k-te Distanz für jeden Punkt
    k_distances = distances[:, k-1]
    k_distances = np.sort(k_distances)
    
    # eps als "Knie" der Kurve schätzen
    # Vereinfachte Heuristik: 90. Perzentil
    eps = np.percentile(k_distances, 90)
    
    return eps


def get_method_info() -> Dict[str, Dict[str, Any]]:
    """Gibt Informationen über verfügbare Clustering-Methoden zurück"""
    return {
        'kmeans': {
            'name': 'K-Means',
            'description': 'Partitionierender Clustering-Algorithmus',
            'pros': ['Schnell', 'Skaliert gut', 'Einfach zu verstehen', 'Kann neue Daten vorhersagen'],
            'cons': ['Anzahl Cluster muss bekannt sein', 'Sphärische Cluster', 'Sensitiv zu Initialisierung'],
            'parameters': {
                'n_clusters': 'Anzahl der Cluster (int)',
                'n_init': 'Anzahl der Initialisierungen (int/str)',
                'max_iter': 'Maximale Iterationen (int)'
            },
            'best_for': 'Sphärische Cluster, bekannte Cluster-Anzahl, große Datensätze'
        },
        'dbscan': {
            'name': 'DBSCAN',
            'description': 'Density-Based Spatial Clustering',
            'pros': ['Findet beliebige Cluster-Formen', 'Erkennt Outlier', 'Keine Cluster-Anzahl nötig'],
            'cons': ['Parameter-sensitiv', 'Schlecht bei unterschiedlichen Dichten', 'Kann keine neuen Daten vorhersagen'],
            'parameters': {
                'eps': 'Maximale Distanz zwischen Nachbarn (float)',
                'min_samples': 'Mindestanzahl Samples pro Cluster (int)',
                'metric': 'Distanz-Metrik (str)'
            },
            'best_for': 'Beliebige Cluster-Formen, Outlier-Erkennung, unbekannte Cluster-Anzahl'
        },
        'hdbscan': {
            'name': 'HDBSCAN',
            'description': 'Hierarchical DBSCAN',
            'pros': ['Findet beliebige Cluster-Formen', 'Erkennt Outlier', 'Robuster als DBSCAN', 'Hierarchische Struktur'],
            'cons': ['Parameter-sensitiv', 'Kann keine neuen Daten vorhersagen', 'Langsamer als DBSCAN'],
            'parameters': {
                'min_cluster_size': 'Mindestgröße eines Clusters (int)',
                'min_samples': 'Mindestanzahl Samples (int)',
                'metric': 'Distanz-Metrik (str)'
            },
            'best_for': 'Komplexe Cluster-Strukturen, hierarchische Daten, Outlier-Erkennung'
        },
        'agglomerative': {
            'name': 'Agglomerative Clustering',
            'description': 'Hierarchischer Clustering-Algorithmus',
            'pros': ['Hierarchische Struktur', 'Deterministisch', 'Kann neue Daten vorhersagen'],
            'cons': ['O(n³) Komplexität', 'Sensitiv zu Outliern', 'Cluster-Anzahl muss bekannt sein'],
            'parameters': {
                'n_clusters': 'Anzahl der Cluster (int)',
                'linkage': 'Verknüpfungs-Methode (str)',
                'metric': 'Distanz-Metrik (str)'
            },
            'best_for': 'Hierarchische Strukturen, kleine bis mittlere Datensätze'
        }
    }


def recommend_method(n_samples: int, n_features: int, use_case: str = 'general') -> str:
    """Empfiehlt eine Clustering-Methode basierend auf Datencharakteristika"""
    
    if use_case == 'outlier_detection':
        return 'dbscan'
    
    elif use_case == 'hierarchical':
        if n_samples < 1000:
            return 'agglomerative'
        else:
            return 'hdbscan'
    
    elif use_case == 'speed':
        return 'kmeans'
    
    elif use_case == 'large_dataset':
        if n_samples > 10000:
            return 'kmeans'
        else:
            return 'dbscan'
    
    else:  # general
        if n_samples < 1000:
            return 'agglomerative'
        elif n_samples < 10000:
            return 'dbscan'
        else:
            return 'kmeans'


def get_optimal_parameters(method: str, n_samples: int, n_features: int) -> Dict[str, Any]:
    """Gibt optimale Parameter für eine Methode basierend auf Datencharakteristika zurück"""
    
    if method == 'kmeans':
        # Heuristik für k: sqrt(n_samples/2)
        k = max(2, int(np.sqrt(n_samples / 2)))
        k = min(k, 20)  # Maximum 20 Cluster
        
        return {
            'n_clusters': k,
            'n_init': 'auto',
            'max_iter': 300
        }
    
    elif method == 'dbscan':
        # eps und min_samples basierend auf Datengröße
        min_samples = max(2, min(10, n_samples // 100))
        
        return {
            'eps': 0.5,
            'min_samples': min_samples,
            'metric': 'euclidean'
        }
    
    elif method == 'hdbscan':
        min_cluster_size = max(2, min(10, n_samples // 50))
        
        return {
            'min_cluster_size': min_cluster_size,
            'min_samples': None,
            'metric': 'euclidean'
        }
    
    elif method == 'agglomerative':
        k = max(2, int(np.sqrt(n_samples / 2)))
        k = min(k, 20)
        
        return {
            'n_clusters': k,
            'linkage': 'ward',
            'metric': 'euclidean'
        }
    
    else:
        return {}


def validate_parameters(method: str, parameters: Dict[str, Any], n_samples: int, n_features: int) -> Dict[str, Any]:
    """Validiert und korrigiert Parameter"""
    validated = parameters.copy()
    
    if method == 'kmeans':
        validated['n_clusters'] = min(validated.get('n_clusters', 8), n_samples)
        validated['n_init'] = validated.get('n_init', 'auto')
    
    elif method == 'dbscan':
        validated['min_samples'] = min(validated.get('min_samples', 5), n_samples)
        validated['eps'] = max(0.01, validated.get('eps', 0.5))
    
    elif method == 'hdbscan':
        validated['min_cluster_size'] = min(validated.get('min_cluster_size', 5), n_samples)
        if validated.get('min_samples') is not None:
            validated['min_samples'] = min(validated['min_samples'], n_samples)
    
    elif method == 'agglomerative':
        validated['n_clusters'] = min(validated.get('n_clusters', 8), n_samples)
    
    return validated
