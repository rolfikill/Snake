"""
Import/Export-Funktionalität für CSV/Parquet, YAML und joblib
"""

import pandas as pd
import numpy as np
import yaml
import joblib
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple
import io


class DataExporter:
    """Klasse für Daten-Export"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Stellt sicher, dass das Output-Verzeichnis existiert"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def export_labeled_data(
        self,
        original_data: pd.DataFrame,
        labels: np.ndarray,
        filename: str = "labeled_data",
        format: str = "csv"
    ) -> str:
        """
        Exportiert die gelabelten Daten
        
        Args:
            original_data: Originale Daten
            labels: Cluster-Labels
            filename: Dateiname (ohne Extension)
            format: Export-Format ('csv' oder 'parquet')
        """
        # DataFrame mit Labels erstellen
        labeled_data = original_data.copy()
        labeled_data['cluster'] = labels
        
        # Dateiname mit Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}"
        
        if format.lower() == 'csv':
            filepath = os.path.join(self.output_dir, f"{full_filename}.csv")
            labeled_data.to_csv(filepath, index=False, encoding='utf-8')
        elif format.lower() == 'parquet':
            filepath = os.path.join(self.output_dir, f"{full_filename}.parquet")
            labeled_data.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filepath
    
    def export_config(
        self,
        config: Dict[str, Any],
        filename: str = "pipeline_config"
    ) -> str:
        """
        Exportiert die Pipeline-Konfiguration
        
        Args:
            config: Pipeline-Konfiguration
            filename: Dateiname (ohne Extension)
        """
        # Timestamp hinzufügen
        config['metadata']['created_at'] = datetime.now().isoformat()
        
        # Dateiname mit Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}"
        
        # YAML exportieren
        yaml_path = os.path.join(self.output_dir, f"{full_filename}.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # JSON exportieren (für Kompatibilität)
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        config_serializable = convert_numpy_types(config)
        
        json_path = os.path.join(self.output_dir, f"{full_filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_serializable, f, indent=2, ensure_ascii=False)
        
        return yaml_path
    
    def export_model(
        self,
        dr_model: Any,
        cluster_model: Any,
        filename: str = "trained_models"
    ) -> str:
        """
        Exportiert die trainierten Modelle
        
        Args:
            dr_model: Dimensionsreduktions-Modell
            cluster_model: Clustering-Modell
            filename: Dateiname (ohne Extension)
        """
        # Modelle-Verzeichnis erstellen
        models_dir = os.path.join(self.output_dir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Dateiname mit Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}"
        
        # Modelle speichern
        model_data = {
            'dr_model': dr_model,
            'cluster_model': cluster_model,
            'timestamp': timestamp,
            'metadata': {
                'dr_method': getattr(dr_model, 'method', 'unknown'),
                'cluster_method': getattr(cluster_model, 'method', 'unknown')
            }
        }
        
        filepath = os.path.join(models_dir, f"{full_filename}.joblib")
        joblib.dump(model_data, filepath)
        
        return filepath
    
    def export_metrics(
        self,
        metrics: Dict[str, Any],
        filename: str = "clustering_metrics"
    ) -> str:
        """
        Exportiert die Clustering-Metriken
        
        Args:
            metrics: Metriken-Dictionary
            filename: Dateiname (ohne Extension)
        """
        # Timestamp hinzufügen
        metrics['metadata'] = {
            'created_at': datetime.now().isoformat(),
            'app_version': '0.1.0'
        }
        
        # Dateiname mit Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}"
        
        # JSON exportieren (numpy types konvertieren)
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        metrics_serializable = convert_numpy_types(metrics)
        
        json_path = os.path.join(self.output_dir, f"{full_filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
        
        return json_path
    
    def export_complete_pipeline(
        self,
        original_data: pd.DataFrame,
        labels: np.ndarray,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        dr_model: Any = None,
        cluster_model: Any = None,
        prefix: str = "clustering_pipeline"
    ) -> Dict[str, str]:
        """
        Exportiert die komplette Pipeline
        
        Args:
            original_data: Originale Daten
            labels: Cluster-Labels
            config: Pipeline-Konfiguration
            metrics: Metriken
            dr_model: Dimensionsreduktions-Modell (optional)
            cluster_model: Clustering-Modell (optional)
            prefix: Präfix für alle Dateien
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exported_files = {}
        
        # Gelabelte Daten exportieren
        labeled_path = self.export_labeled_data(
            original_data, labels, f"{prefix}_data", "csv"
        )
        exported_files['labeled_data'] = labeled_path
        
        # Konfiguration exportieren
        config_path = self.export_config(config, f"{prefix}_config")
        exported_files['config'] = config_path
        
        # Metriken exportieren
        metrics_path = self.export_metrics(metrics, f"{prefix}_metrics")
        exported_files['metrics'] = metrics_path
        
        # Modelle exportieren (falls vorhanden)
        if dr_model is not None and cluster_model is not None:
            model_path = self.export_model(dr_model, cluster_model, f"{prefix}_models")
            exported_files['models'] = model_path
        
        return exported_files


class DataImporter:
    """Klasse für Daten-Import"""
    
    def __init__(self):
        pass
    
    def import_config(self, filepath: str) -> Dict[str, Any]:
        """
        Importiert eine Pipeline-Konfiguration
        
        Args:
            filepath: Pfad zur Konfigurationsdatei
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {filepath}")
        
        return config
    
    def import_model(self, filepath: str) -> Dict[str, Any]:
        """
        Importiert trainierte Modelle
        
        Args:
            filepath: Pfad zur Modell-Datei
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        return model_data
    
    def import_metrics(self, filepath: str) -> Dict[str, Any]:
        """
        Importiert Metriken
        
        Args:
            filepath: Pfad zur Metriken-Datei
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metrics file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        return metrics


def create_streamlit_download_button(
    data: Union[pd.DataFrame, Dict[str, Any], str],
    filename: str,
    mime_type: str,
    label: str = "Download"
) -> bytes:
    """
    Erstellt Download-Button für Streamlit
    
    Args:
        data: Daten zum Download
        filename: Dateiname
        mime_type: MIME-Type
        label: Button-Label
    """
    if isinstance(data, pd.DataFrame):
        if filename.endswith('.csv'):
            return data.to_csv(index=False).encode('utf-8')
        elif filename.endswith('.parquet'):
            buffer = io.BytesIO()
            data.to_parquet(buffer, index=False)
            return buffer.getvalue()
    elif isinstance(data, dict):
        if filename.endswith('.json'):
            return json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            return yaml.dump(data, default_flow_style=False, allow_unicode=True).encode('utf-8')
    elif isinstance(data, str):
        return data.encode('utf-8')
    
    raise ValueError(f"Unsupported data type for download: {type(data)}")


def get_file_size_mb(filepath: str) -> float:
    """Gibt die Dateigröße in MB zurück"""
    if not os.path.exists(filepath):
        return 0.0
    
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)


def cleanup_old_files(directory: str, max_files: int = 10, pattern: str = "*"):
    """
    Bereinigt alte Dateien in einem Verzeichnis
    
    Args:
        directory: Verzeichnis
        max_files: Maximale Anzahl Dateien
        pattern: Dateimuster
    """
    if not os.path.exists(directory):
        return
    
    import glob
    
    files = glob.glob(os.path.join(directory, pattern))
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Alte Dateien löschen
    for file in files[max_files:]:
        try:
            os.remove(file)
        except Exception:
            pass


def create_backup(filepath: str, backup_dir: str = "backups") -> str:
    """
    Erstellt ein Backup einer Datei
    
    Args:
        filepath: Pfad zur Datei
        backup_dir: Backup-Verzeichnis
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    filename = os.path.basename(filepath)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{timestamp}_{filename}"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    import shutil
    shutil.copy2(filepath, backup_path)
    
    return backup_path


def validate_export_data(
    original_data: pd.DataFrame,
    labels: np.ndarray
) -> Tuple[bool, str]:
    """
    Validiert Daten vor dem Export
    
    Args:
        original_data: Originale Daten
        labels: Cluster-Labels
    
    Returns:
        (is_valid, error_message)
    """
    if original_data is None:
        return False, "Keine Original-Daten vorhanden"
    
    if labels is None:
        return False, "Keine Cluster-Labels vorhanden"
    
    if len(original_data) != len(labels):
        return False, f"Länge der Daten ({len(original_data)}) stimmt nicht mit Labels ({len(labels)}) überein"
    
    if len(original_data) == 0:
        return False, "Daten sind leer"
    
    return True, ""


def get_export_summary(exported_files: Dict[str, str]) -> Dict[str, Any]:
    """
    Erstellt eine Zusammenfassung der exportierten Dateien
    
    Args:
        exported_files: Dictionary mit exportierten Dateien
    """
    summary = {
        'total_files': len(exported_files),
        'files': {},
        'total_size_mb': 0.0
    }
    
    for file_type, filepath in exported_files.items():
        if os.path.exists(filepath):
            size_mb = get_file_size_mb(filepath)
            summary['files'][file_type] = {
                'path': filepath,
                'size_mb': size_mb,
                'filename': os.path.basename(filepath)
            }
            summary['total_size_mb'] += size_mb
    
    return summary


def format_file_size(size_mb: float) -> str:
    """Formatiert Dateigröße für Anzeige"""
    if size_mb < 1:
        return f"{size_mb * 1024:.1f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb / 1024:.1f} GB"


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
