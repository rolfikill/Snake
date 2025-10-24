"""
SessionState und Seed-Management
"""

import streamlit as st
import numpy as np
import random
from typing import Dict, Any, Optional, Union
import hashlib


class SessionState:
    """Klasse für SessionState-Management"""
    
    def __init__(self):
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialisiert den SessionState"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = None
        
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        
        if 'column_analysis' not in st.session_state:
            st.session_state.column_analysis = None
        
        if 'dr_model' not in st.session_state:
            st.session_state.dr_model = None
        
        if 'dr_data' not in st.session_state:
            st.session_state.dr_data = None
        
        if 'cluster_model' not in st.session_state:
            st.session_state.cluster_model = None
        
        if 'cluster_labels' not in st.session_state:
            st.session_state.cluster_labels = None
        
        if 'metrics' not in st.session_state:
            st.session_state.metrics = None
        
        if 'random_seed' not in st.session_state:
            st.session_state.random_seed = 42
        
        if 'pipeline_config' not in st.session_state:
            st.session_state.pipeline_config = {}
        
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 'upload'
        
        if 'file_info' not in st.session_state:
            st.session_state.file_info = {}
    
    def set_data(self, data, file_info: Dict[str, Any] = None):
        """Setzt die geladenen Daten"""
        st.session_state.raw_data = data
        st.session_state.data_loaded = True
        if file_info:
            st.session_state.file_info = file_info
    
    def get_data(self):
        """Gibt die geladenen Daten zurück"""
        return st.session_state.raw_data
    
    def set_processed_data(self, data):
        """Setzt die verarbeiteten Daten"""
        st.session_state.processed_data = data
    
    def get_processed_data(self):
        """Gibt die verarbeiteten Daten zurück"""
        return st.session_state.processed_data
    
    def set_column_analysis(self, analysis):
        """Setzt die Spalten-Analyse"""
        st.session_state.column_analysis = analysis
    
    def get_column_analysis(self):
        """Gibt die Spalten-Analyse zurück"""
        return st.session_state.column_analysis
    
    def set_dr_model(self, model, data):
        """Setzt das Dimensionsreduktions-Modell und die Daten"""
        st.session_state.dr_model = model
        st.session_state.dr_data = data
    
    def get_dr_model(self):
        """Gibt das Dimensionsreduktions-Modell zurück"""
        return st.session_state.dr_model
    
    def get_dr_data(self):
        """Gibt die dimensionsreduzierten Daten zurück"""
        return st.session_state.dr_data
    
    def set_cluster_model(self, model, labels):
        """Setzt das Clustering-Modell und die Labels"""
        st.session_state.cluster_model = model
        st.session_state.cluster_labels = labels
    
    def get_cluster_model(self):
        """Gibt das Clustering-Modell zurück"""
        return st.session_state.cluster_model
    
    def get_cluster_labels(self):
        """Gibt die Cluster-Labels zurück"""
        return st.session_state.cluster_labels
    
    def set_metrics(self, metrics):
        """Setzt die Metriken"""
        st.session_state.metrics = metrics
    
    def get_metrics(self):
        """Gibt die Metriken zurück"""
        return st.session_state.metrics
    
    def set_random_seed(self, seed: int):
        """Setzt den Random Seed"""
        st.session_state.random_seed = seed
        self._set_global_seed(seed)
    
    def get_random_seed(self) -> int:
        """Gibt den aktuellen Random Seed zurück"""
        return st.session_state.random_seed
    
    def set_pipeline_config(self, config: Dict[str, Any]):
        """Setzt die Pipeline-Konfiguration"""
        st.session_state.pipeline_config = config
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Gibt die Pipeline-Konfiguration zurück"""
        return st.session_state.pipeline_config
    
    def set_current_step(self, step: str):
        """Setzt den aktuellen Schritt"""
        st.session_state.current_step = step
    
    def get_current_step(self) -> str:
        """Gibt den aktuellen Schritt zurück"""
        return st.session_state.current_step
    
    def get_file_info(self) -> Dict[str, Any]:
        """Gibt die Datei-Informationen zurück"""
        return st.session_state.file_info
    
    def _set_global_seed(self, seed: int):
        """Setzt den globalen Random Seed"""
        np.random.seed(seed)
        random.seed(seed)
    
    def reset_state(self):
        """Setzt den gesamten State zurück"""
        keys_to_reset = [
            'data_loaded', 'raw_data', 'processed_data', 'column_analysis',
            'dr_model', 'dr_data', 'cluster_model', 'cluster_labels',
            'metrics', 'pipeline_config', 'current_step', 'file_info'
        ]
        
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        
        self._initialize_state()
    
    def is_data_loaded(self) -> bool:
        """Prüft ob Daten geladen sind"""
        return st.session_state.data_loaded
    
    def is_processed(self) -> bool:
        """Prüft ob Daten verarbeitet sind"""
        return st.session_state.processed_data is not None
    
    def is_dr_computed(self) -> bool:
        """Prüft ob Dimensionsreduktion durchgeführt wurde"""
        return st.session_state.dr_data is not None
    
    def is_clustered(self) -> bool:
        """Prüft ob Clustering durchgeführt wurde"""
        return st.session_state.cluster_labels is not None
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung des aktuellen States zurück"""
        return {
            'data_loaded': self.is_data_loaded(),
            'data_shape': st.session_state.raw_data.shape if st.session_state.raw_data is not None else None,
            'processed': self.is_processed(),
            'processed_shape': st.session_state.processed_data.shape if st.session_state.processed_data is not None else None,
            'dr_computed': self.is_dr_computed(),
            'dr_shape': st.session_state.dr_data.shape if st.session_state.dr_data is not None else None,
            'clustered': self.is_clustered(),
            'n_clusters': len(np.unique(st.session_state.cluster_labels)) if st.session_state.cluster_labels is not None else None,
            'random_seed': self.get_random_seed(),
            'current_step': self.get_current_step()
        }


class SeedManager:
    """Klasse für Seed-Management"""
    
    def __init__(self, default_seed: int = 42):
        self.default_seed = default_seed
        self.current_seed = default_seed
    
    def set_seed(self, seed: Union[int, str, None]):
        """
        Setzt den Random Seed
        
        Args:
            seed: Integer, String oder None
        """
        if seed is None:
            seed = self.default_seed
        elif isinstance(seed, str):
            # String zu Integer konvertieren (Hash)
            seed = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32)
        elif not isinstance(seed, int):
            seed = int(seed)
        
        self.current_seed = seed
        self._apply_seed(seed)
        return seed
    
    def _apply_seed(self, seed: int):
        """Wendet den Seed auf alle relevanten Bibliotheken an"""
        np.random.seed(seed)
        random.seed(seed)
        
        # Zusätzliche Seeds für spezifische Bibliotheken
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
    
    def get_seed(self) -> int:
        """Gibt den aktuellen Seed zurück"""
        return self.current_seed
    
    def generate_seed(self) -> int:
        """Generiert einen neuen zufälligen Seed"""
        seed = random.randint(0, 2**32 - 1)
        self.set_seed(seed)
        return seed
    
    def reset_to_default(self):
        """Setzt den Seed auf den Standardwert zurück"""
        self.set_seed(self.default_seed)


def create_pipeline_config(
    preprocessing: Dict[str, Any],
    dr_method: str,
    dr_params: Dict[str, Any],
    cluster_method: str,
    cluster_params: Dict[str, Any],
    random_seed: int
) -> Dict[str, Any]:
    """
    Erstellt eine Pipeline-Konfiguration
    
    Args:
        preprocessing: Preprocessing-Parameter
        dr_method: Dimensionsreduktions-Methode
        dr_params: Dimensionsreduktions-Parameter
        cluster_method: Clustering-Methode
        cluster_params: Clustering-Parameter
        random_seed: Random Seed
    """
    return {
        'version': '1.0',
        'random_seed': random_seed,
        'preprocessing': preprocessing,
        'dimensionality_reduction': {
            'method': dr_method,
            'parameters': dr_params
        },
        'clustering': {
            'method': cluster_method,
            'parameters': cluster_params
        },
        'metadata': {
            'created_at': None,  # Wird beim Speichern gesetzt
            'app_version': '0.1.0'
        }
    }


def validate_pipeline_config(config: Dict[str, Any]) -> bool:
    """
    Validiert eine Pipeline-Konfiguration
    
    Args:
        config: Pipeline-Konfiguration
    """
    required_keys = [
        'version', 'random_seed', 'preprocessing',
        'dimensionality_reduction', 'clustering'
    ]
    
    for key in required_keys:
        if key not in config:
            return False
    
    # Dimensionsreduktion validieren
    dr_config = config['dimensionality_reduction']
    if 'method' not in dr_config or 'parameters' not in dr_config:
        return False
    
    # Clustering validieren
    cluster_config = config['clustering']
    if 'method' not in cluster_config or 'parameters' not in cluster_config:
        return False
    
    return True


def get_step_info() -> Dict[str, Dict[str, str]]:
    """Gibt Informationen über die Pipeline-Schritte zurück"""
    return {
        'upload': {
            'name': 'Daten-Upload',
            'description': 'CSV oder Parquet Datei hochladen',
            'icon': '📁'
        },
        'preprocessing': {
            'name': 'Vorverarbeitung',
            'description': 'Daten bereinigen und vorbereiten',
            'icon': '🔧'
        },
        'dimensionality_reduction': {
            'name': 'Dimensionsreduktion',
            'description': 'PCA, UMAP oder t-SNE anwenden',
            'icon': '📉'
        },
        'clustering': {
            'name': 'Clustering',
            'description': 'KMeans, DBSCAN, HDBSCAN oder Agglomerative',
            'icon': '🎯'
        },
        'results': {
            'name': 'Ergebnisse',
            'description': 'Visualisierung und Metriken',
            'icon': '📊'
        }
    }


def get_next_step(current_step: str) -> Optional[str]:
    """Gibt den nächsten Schritt zurück"""
    steps = ['upload', 'preprocessing', 'dimensionality_reduction', 'clustering', 'results']
    
    try:
        current_index = steps.index(current_step)
        if current_index < len(steps) - 1:
            return steps[current_index + 1]
    except ValueError:
        pass
    
    return None


def get_previous_step(current_step: str) -> Optional[str]:
    """Gibt den vorherigen Schritt zurück"""
    steps = ['upload', 'preprocessing', 'dimensionality_reduction', 'clustering', 'results']
    
    try:
        current_index = steps.index(current_step)
        if current_index > 0:
            return steps[current_index - 1]
    except ValueError:
        pass
    
    return None


def can_proceed_to_step(step: str, state: SessionState) -> bool:
    """
    Prüft ob zu einem bestimmten Schritt gewechselt werden kann
    
    Args:
        step: Ziel-Schritt
        state: SessionState
    """
    if step == 'upload':
        return True
    elif step == 'preprocessing':
        return state.is_data_loaded()
    elif step == 'dimensionality_reduction':
        return state.is_processed()
    elif step == 'clustering':
        return state.is_dr_computed()
    elif step == 'results':
        return state.is_clustered()
    
    return False
