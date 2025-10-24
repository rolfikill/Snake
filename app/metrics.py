"""
Metriken-Modul für Clustering-Qualität
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
import warnings

# Warnungen für Metriken unterdrücken
warnings.filterwarnings('ignore', category=UserWarning)


class ClusteringMetrics:
    """Klasse für Clustering-Metriken"""
    
    def __init__(self):
        self.metrics = {}
        self.is_computed = False
        
    def compute_all_metrics(
        self, 
        X: np.ndarray, 
        labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Berechnet alle verfügbaren Metriken
        
        Args:
            X: Eingabedaten
            labels: Cluster-Labels
            true_labels: Wahre Labels (optional, für externe Metriken)
        """
        self.metrics = {}
        
        # Interne Metriken (benötigen keine wahren Labels)
        self._compute_internal_metrics(X, labels)
        
        # Externe Metriken (benötigen wahre Labels)
        if true_labels is not None:
            self._compute_external_metrics(labels, true_labels)
        
        # Cluster-Statistiken
        self._compute_cluster_statistics(labels)
        
        self.is_computed = True
        return self.metrics
    
    def _compute_internal_metrics(self, X: np.ndarray, labels: np.ndarray):
        """Berechnet interne Metriken"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Nur berechnen wenn mindestens 2 Cluster vorhanden sind
        if n_clusters < 2:
            self.metrics['silhouette_score'] = None
            self.metrics['davies_bouldin_score'] = None
            self.metrics['calinski_harabasz_score'] = None
            return
        
        # Silhouette Score
        try:
            # Nur berechnen wenn nicht alle Punkte in einem Cluster sind
            if len(unique_labels) > 1 and not np.all(labels == labels[0]):
                self.metrics['silhouette_score'] = silhouette_score(X, labels)
            else:
                self.metrics['silhouette_score'] = None
        except Exception:
            self.metrics['silhouette_score'] = None
        
        # Davies-Bouldin Score
        try:
            if n_clusters > 1:
                self.metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            else:
                self.metrics['davies_bouldin_score'] = None
        except Exception:
            self.metrics['davies_bouldin_score'] = None
        
        # Calinski-Harabasz Score
        try:
            if n_clusters > 1:
                self.metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            else:
                self.metrics['calinski_harabasz_score'] = None
        except Exception:
            self.metrics['calinski_harabasz_score'] = None
    
    def _compute_external_metrics(self, labels: np.ndarray, true_labels: np.ndarray):
        """Berechnet externe Metriken"""
        try:
            self.metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, labels)
        except Exception:
            self.metrics['adjusted_rand_score'] = None
        
        try:
            self.metrics['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, labels)
        except Exception:
            self.metrics['normalized_mutual_info_score'] = None
        
        try:
            self.metrics['homogeneity_score'] = homogeneity_score(true_labels, labels)
        except Exception:
            self.metrics['homogeneity_score'] = None
        
        try:
            self.metrics['completeness_score'] = completeness_score(true_labels, labels)
        except Exception:
            self.metrics['completeness_score'] = None
        
        try:
            self.metrics['v_measure_score'] = v_measure_score(true_labels, labels)
        except Exception:
            self.metrics['v_measure_score'] = None
    
    def _compute_cluster_statistics(self, labels: np.ndarray):
        """Berechnet Cluster-Statistiken"""
        unique_labels = np.unique(labels)
        
        # Grundlegende Statistiken
        self.metrics['n_clusters'] = len(unique_labels)
        self.metrics['n_points'] = len(labels)
        
        # Cluster-Größen
        cluster_sizes = {}
        outlier_count = 0
        
        for label in unique_labels:
            count = np.sum(labels == label)
            if label == -1:  # Noise/Outlier
                outlier_count = count
                cluster_sizes['outlier'] = count
            else:
                cluster_sizes[f'cluster_{label}'] = count
        
        self.metrics['cluster_sizes'] = cluster_sizes
        self.metrics['outlier_count'] = outlier_count
        self.metrics['outlier_percentage'] = (outlier_count / len(labels)) * 100 if len(labels) > 0 else 0
        
        # Cluster-Größen-Statistiken
        if cluster_sizes:
            sizes = [size for label, size in cluster_sizes.items() if label != 'outlier']
            if sizes:
                self.metrics['min_cluster_size'] = min(sizes)
                self.metrics['max_cluster_size'] = max(sizes)
                self.metrics['mean_cluster_size'] = np.mean(sizes)
                self.metrics['std_cluster_size'] = np.std(sizes)
            else:
                self.metrics['min_cluster_size'] = 0
                self.metrics['max_cluster_size'] = 0
                self.metrics['mean_cluster_size'] = 0
                self.metrics['std_cluster_size'] = 0
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung der Metriken zurück"""
        if not self.is_computed:
            return {}
        
        summary = {
            'internal_metrics': {},
            'external_metrics': {},
            'cluster_statistics': {}
        }
        
        # Interne Metriken
        internal_keys = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
        for key in internal_keys:
            if key in self.metrics and self.metrics[key] is not None:
                summary['internal_metrics'][key] = {
                    'value': self.metrics[key],
                    'interpretation': self._interpret_metric(key, self.metrics[key])
                }
        
        # Externe Metriken
        external_keys = [
            'adjusted_rand_score', 'normalized_mutual_info_score', 
            'homogeneity_score', 'completeness_score', 'v_measure_score'
        ]
        for key in external_keys:
            if key in self.metrics and self.metrics[key] is not None:
                summary['external_metrics'][key] = {
                    'value': self.metrics[key],
                    'interpretation': self._interpret_metric(key, self.metrics[key])
                }
        
        # Cluster-Statistiken
        stats_keys = [
            'n_clusters', 'n_points', 'outlier_count', 'outlier_percentage',
            'min_cluster_size', 'max_cluster_size', 'mean_cluster_size', 'std_cluster_size'
        ]
        for key in stats_keys:
            if key in self.metrics:
                summary['cluster_statistics'][key] = self.metrics[key]
        
        return summary
    
    def _interpret_metric(self, metric_name: str, value: float) -> str:
        """Interpretiert eine Metrik"""
        interpretations = {
            'silhouette_score': {
                'range': (0, 1),
                'good': '> 0.5',
                'description': 'Misst die Trennung zwischen Clustern. Höhere Werte sind besser.'
            },
            'davies_bouldin_score': {
                'range': (0, float('inf')),
                'good': '< 1.0',
                'description': 'Misst die Kompaktheit und Trennung. Niedrigere Werte sind besser.'
            },
            'calinski_harabasz_score': {
                'range': (0, float('inf')),
                'good': '> 100',
                'description': 'Verhältnis von Between-Cluster zu Within-Cluster Varianz. Höhere Werte sind besser.'
            },
            'adjusted_rand_score': {
                'range': (-1, 1),
                'good': '> 0.5',
                'description': 'Ähnlichkeit zu wahren Labels. Höhere Werte sind besser.'
            },
            'normalized_mutual_info_score': {
                'range': (0, 1),
                'good': '> 0.5',
                'description': 'Normalisierte gegenseitige Information. Höhere Werte sind besser.'
            },
            'homogeneity_score': {
                'range': (0, 1),
                'good': '> 0.5',
                'description': 'Jeder Cluster enthält nur Mitglieder einer Klasse. Höhere Werte sind besser.'
            },
            'completeness_score': {
                'range': (0, 1),
                'good': '> 0.5',
                'description': 'Alle Mitglieder einer Klasse sind im selben Cluster. Höhere Werte sind besser.'
            },
            'v_measure_score': {
                'range': (0, 1),
                'good': '> 0.5',
                'description': 'Harmonisches Mittel von Homogenität und Vollständigkeit. Höhere Werte sind besser.'
            }
        }
        
        if metric_name in interpretations:
            info = interpretations[metric_name]
            return f"{info['description']} Bereich: {info['range']}, Gut: {info['good']}"
        
        return "Keine Interpretation verfügbar"


def compute_silhouette_analysis(
    X: np.ndarray, 
    labels: np.ndarray,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Führt detaillierte Silhouette-Analyse durch
    
    Args:
        X: Eingabedaten
        labels: Cluster-Labels
        sample_size: Stichprobengröße für große Datensätze
    """
    from sklearn.metrics import silhouette_samples
    
    if sample_size and len(X) > sample_size:
        # Stichprobe für große Datensätze
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels
    
    try:
        silhouette_vals = silhouette_samples(X_sample, labels_sample)
        
        # Silhouette-Werte pro Cluster
        unique_labels = np.unique(labels_sample)
        cluster_silhouettes = {}
        
        for label in unique_labels:
            if label != -1:  # Ignoriere Noise
                mask = labels_sample == label
                cluster_silhouettes[f'cluster_{label}'] = {
                    'mean': np.mean(silhouette_vals[mask]),
                    'std': np.std(silhouette_vals[mask]),
                    'min': np.min(silhouette_vals[mask]),
                    'max': np.max(silhouette_vals[mask])
                }
        
        return {
            'overall_silhouette': np.mean(silhouette_vals),
            'cluster_silhouettes': cluster_silhouettes,
            'silhouette_values': silhouette_vals,
            'sample_size': len(X_sample)
        }
        
    except Exception as e:
        return {'error': str(e)}


def get_metric_descriptions() -> Dict[str, Dict[str, str]]:
    """Gibt Beschreibungen aller Metriken zurück"""
    return {
        'silhouette_score': {
            'name': 'Silhouette Score',
            'description': 'Misst die Qualität der Cluster-Trennung basierend auf der Distanz zwischen Clustern und der Kompaktheit innerhalb von Clustern.',
            'range': '[-1, 1]',
            'interpretation': 'Höhere Werte bedeuten bessere Cluster-Trennung. > 0.5: starke Struktur, 0.2-0.5: vernünftige Struktur, < 0.2: schwache Struktur'
        },
        'davies_bouldin_score': {
            'name': 'Davies-Bouldin Score',
            'description': 'Misst das Verhältnis von Within-Cluster zu Between-Cluster Distanzen.',
            'range': '[0, ∞)',
            'interpretation': 'Niedrigere Werte bedeuten bessere Cluster. < 1.0: gute Cluster, 1.0-2.0: moderate Cluster, > 2.0: schlechte Cluster'
        },
        'calinski_harabasz_score': {
            'name': 'Calinski-Harabasz Score',
            'description': 'Misst das Verhältnis von Between-Cluster zu Within-Cluster Varianz.',
            'range': '[0, ∞)',
            'interpretation': 'Höhere Werte bedeuten bessere Cluster. > 100: gute Cluster, 50-100: moderate Cluster, < 50: schlechte Cluster'
        },
        'adjusted_rand_score': {
            'name': 'Adjusted Rand Score',
            'description': 'Misst die Ähnlichkeit zwischen vorhergesagten und wahren Clustern.',
            'range': '[-1, 1]',
            'interpretation': 'Höhere Werte bedeuten bessere Übereinstimmung. > 0.5: gute Übereinstimmung, 0.2-0.5: moderate Übereinstimmung, < 0.2: schlechte Übereinstimmung'
        },
        'normalized_mutual_info_score': {
            'name': 'Normalized Mutual Information',
            'description': 'Misst die normalisierte gegenseitige Information zwischen Clustern.',
            'range': '[0, 1]',
            'interpretation': 'Höhere Werte bedeuten bessere Übereinstimmung. > 0.5: gute Übereinstimmung, 0.2-0.5: moderate Übereinstimmung, < 0.2: schlechte Übereinstimmung'
        }
    }


def create_metrics_dataframe(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Erstellt einen DataFrame mit den Metriken für die Anzeige"""
    data = []
    
    # Interne Metriken
    internal_metrics = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
    for metric in internal_metrics:
        if metric in metrics and metrics[metric] is not None:
            data.append({
                'Kategorie': 'Interne Metriken',
                'Metrik': metric.replace('_', ' ').title(),
                'Wert': f"{metrics[metric]:.4f}",
                'Typ': 'Interne'
            })
    
    # Externe Metriken
    external_metrics = [
        'adjusted_rand_score', 'normalized_mutual_info_score',
        'homogeneity_score', 'completeness_score', 'v_measure_score'
    ]
    for metric in external_metrics:
        if metric in metrics and metrics[metric] is not None:
            data.append({
                'Kategorie': 'Externe Metriken',
                'Metrik': metric.replace('_', ' ').title(),
                'Wert': f"{metrics[metric]:.4f}",
                'Typ': 'Externe'
            })
    
    # Cluster-Statistiken
    stats_metrics = [
        'n_clusters', 'outlier_count', 'outlier_percentage',
        'min_cluster_size', 'max_cluster_size', 'mean_cluster_size'
    ]
    for metric in stats_metrics:
        if metric in metrics:
            value = metrics[metric]
            if isinstance(value, float):
                value = f"{value:.2f}"
            data.append({
                'Kategorie': 'Cluster-Statistiken',
                'Metrik': metric.replace('_', ' ').title(),
                'Wert': str(value),
                'Typ': 'Statistik'
            })
    
    return pd.DataFrame(data)
