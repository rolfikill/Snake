"""
Dimensionsreduktions-Modul für PCA, UMAP und t-SNE
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
import umap
import warnings

# UMAP Warnungen unterdrücken
warnings.filterwarnings('ignore', category=UserWarning, module='umap')


class DimensionalityReducer:
    """Klasse für Dimensionsreduktion"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.method = None
        self.n_components = None
        self.is_fitted = False
        
    def fit_transform(
        self, 
        X: np.ndarray, 
        method: str = 'pca',
        n_components: int = 2,
        **kwargs
    ) -> np.ndarray:
        """
        Führt Dimensionsreduktion durch
        
        Args:
            X: Eingabedaten
            method: Methode ('pca', 'umap', 'tsne')
            n_components: Anzahl der Komponenten
            **kwargs: Zusätzliche Parameter für die jeweilige Methode
        """
        self.method = method
        self.n_components = min(n_components, X.shape[1])
        
        if method == 'pca':
            return self._fit_transform_pca(X, **kwargs)
        elif method == 'umap':
            return self._fit_transform_umap(X, **kwargs)
        elif method == 'tsne':
            return self._fit_transform_tsne(X, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _fit_transform_pca(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """PCA Dimensionsreduktion"""
        # Für große Datensätze IncrementalPCA verwenden
        if X.shape[0] > 10000:
            self.model = IncrementalPCA(
                n_components=self.n_components,
                random_state=self.random_state,
                **kwargs
            )
        else:
            self.model = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
                **kwargs
            )
        
        self.is_fitted = True
        return self.model.fit_transform(X)
    
    def _fit_transform_umap(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """UMAP Dimensionsreduktion"""
        # Standard-Parameter
        umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean',
            'n_components': self.n_components,
            'random_state': self.random_state,
            'verbose': False
        }
        
        # Benutzer-Parameter überschreiben
        umap_params.update(kwargs)
        
        self.model = umap.UMAP(**umap_params)
        self.is_fitted = True
        return self.model.fit_transform(X)
    
    def _fit_transform_tsne(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """t-SNE Dimensionsreduktion"""
        # Standard-Parameter
        tsne_params = {
            'perplexity': 30,
            'learning_rate': 200,
            'n_iter': 1000,
            'init': 'pca',
            'n_components': self.n_components,
            'random_state': self.random_state,
            'verbose': 0
        }
        
        # Benutzer-Parameter überschreiben
        tsne_params.update(kwargs)
        
        # Perplexity anpassen wenn nötig
        if tsne_params['perplexity'] >= X.shape[0]:
            tsne_params['perplexity'] = max(5, X.shape[0] // 3)
        
        self.model = TSNE(**tsne_params)
        self.is_fitted = True
        return self.model.fit_transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transformiert neue Daten mit dem trainierten Modell"""
        if not self.is_fitted:
            raise ValueError("Model muss zuerst gefittet werden")
        
        if self.method == 'pca':
            return self.model.transform(X)
        else:
            # UMAP und t-SNE können keine neuen Daten transformieren
            raise ValueError(f"Transform für {self.method} nicht unterstützt")
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Gibt erklärte Varianz für PCA zurück"""
        if self.method == 'pca' and self.is_fitted:
            return self.model.explained_variance_ratio_
        return None
    
    def get_cumulative_variance_ratio(self) -> Optional[np.ndarray]:
        """Gibt kumulative erklärte Varianz für PCA zurück"""
        if self.method == 'pca' and self.is_fitted:
            return np.cumsum(self.model.explained_variance_ratio_)
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Informationen über das Modell zurück"""
        info = {
            'method': self.method,
            'n_components': self.n_components,
            'is_fitted': self.is_fitted
        }
        
        if self.method == 'pca' and self.is_fitted:
            info['explained_variance_ratio'] = self.get_explained_variance_ratio()
            info['cumulative_variance_ratio'] = self.get_cumulative_variance_ratio()
            info['total_variance_explained'] = np.sum(self.get_explained_variance_ratio())
        
        return info


def get_method_info() -> Dict[str, Dict[str, Any]]:
    """Gibt Informationen über verfügbare Methoden zurück"""
    return {
        'pca': {
            'name': 'Principal Component Analysis',
            'description': 'Lineare Dimensionsreduktion basierend auf Hauptkomponenten',
            'pros': ['Schnell', 'Reproduzierbar', 'Interpretierbar', 'Kann neue Daten transformieren'],
            'cons': ['Nur lineare Beziehungen', 'Kann komplexe Strukturen verlieren'],
            'parameters': {
                'n_components': 'Anzahl der Komponenten (int)',
                'whiten': 'Normalisierung der Komponenten (bool)',
                'svd_solver': 'SVD-Löser (str)'
            },
            'best_for': 'Lineare Daten, große Datensätze, wenn Reproduzierbarkeit wichtig ist'
        },
        'umap': {
            'name': 'Uniform Manifold Approximation and Projection',
            'description': 'Nicht-lineare Dimensionsreduktion basierend auf Riemannscher Geometrie',
            'pros': ['Behält lokale und globale Struktur', 'Schnell', 'Skaliert gut'],
            'cons': ['Parameter-sensitiv', 'Kann neue Daten nicht transformieren'],
            'parameters': {
                'n_neighbors': 'Anzahl der Nachbarn für lokale Struktur (int)',
                'min_dist': 'Minimale Distanz zwischen Punkten (float)',
                'metric': 'Distanz-Metrik (str)',
                'n_components': 'Anzahl der Komponenten (int)'
            },
            'best_for': 'Komplexe nicht-lineare Strukturen, große Datensätze'
        },
        'tsne': {
            'name': 't-Distributed Stochastic Neighbor Embedding',
            'description': 'Nicht-lineare Dimensionsreduktion für Visualisierung',
            'pros': ['Sehr gute Visualisierung', 'Behält lokale Struktur'],
            'cons': ['Langsam', 'Parameter-sensitiv', 'Kann neue Daten nicht transformieren', 'Verliert globale Struktur'],
            'parameters': {
                'perplexity': 'Anzahl der effektiven Nachbarn (float)',
                'learning_rate': 'Lernrate (float)',
                'n_iter': 'Anzahl der Iterationen (int)',
                'n_components': 'Anzahl der Komponenten (int)'
            },
            'best_for': 'Visualisierung, kleine bis mittlere Datensätze'
        }
    }


def recommend_method(n_samples: int, n_features: int, use_case: str = 'general') -> str:
    """Empfiehlt eine Dimensionsreduktions-Methode basierend auf Datencharakteristika"""
    
    if use_case == 'visualization':
        if n_samples < 1000:
            return 'tsne'
        else:
            return 'umap'
    
    elif use_case == 'speed':
        return 'pca'
    
    elif use_case == 'large_dataset':
        if n_samples > 10000:
            return 'pca'
        else:
            return 'umap'
    
    else:  # general
        if n_samples < 1000:
            return 'tsne'
        elif n_samples < 10000:
            return 'umap'
        else:
            return 'pca'


def get_optimal_parameters(method: str, n_samples: int, n_features: int) -> Dict[str, Any]:
    """Gibt optimale Parameter für eine Methode basierend auf Datencharakteristika zurück"""
    
    if method == 'pca':
        return {
            'n_components': min(50, n_features),
            'whiten': False,
            'svd_solver': 'auto'
        }
    
    elif method == 'umap':
        # n_neighbors basierend auf Datengröße anpassen
        n_neighbors = min(15, max(5, n_samples // 100))
        
        return {
            'n_neighbors': n_neighbors,
            'min_dist': 0.1,
            'metric': 'euclidean',
            'n_components': 2
        }
    
    elif method == 'tsne':
        # Perplexity basierend auf Datengröße anpassen
        perplexity = min(30, max(5, n_samples // 4))
        
        return {
            'perplexity': perplexity,
            'learning_rate': 200,
            'n_iter': 1000,
            'n_components': 2
        }
    
    else:
        return {}


def validate_parameters(method: str, parameters: Dict[str, Any], n_samples: int, n_features: int) -> Dict[str, Any]:
    """Validiert und korrigiert Parameter"""
    validated = parameters.copy()
    
    if method == 'pca':
        validated['n_components'] = min(validated.get('n_components', 2), n_features)
    
    elif method == 'umap':
        validated['n_neighbors'] = min(validated.get('n_neighbors', 15), n_samples - 1)
        validated['n_components'] = min(validated.get('n_components', 2), n_features)
    
    elif method == 'tsne':
        validated['perplexity'] = min(validated.get('perplexity', 30), n_samples // 3)
        validated['n_components'] = min(validated.get('n_components', 2), n_features)
    
    return validated
