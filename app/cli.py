"""
CLI-Modus für Batch-Runs ohne UI
"""

import click
import pandas as pd
import numpy as np
import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Import der App-Module
from data import DataProcessor, create_sample_data
from dr import DimensionalityReducer, get_optimal_parameters as get_dr_optimal_params
from cluster import Clusterer, get_optimal_parameters as get_cluster_optimal_params
from metrics import ClusteringMetrics
from io_utils import DataExporter, create_pipeline_config
from state import SeedManager


@click.group()
@click.version_option(version='0.1.0')
def main():
    """Data Clustering App - CLI für Batch-Runs"""
    pass


@main.command()
@click.option('--input', '-i', required=True, help='Eingabe-Datei (CSV oder Parquet)')
@click.option('--output', '-o', default='output', help='Output-Verzeichnis')
@click.option('--algo', '-a', default='kmeans', 
              type=click.Choice(['kmeans', 'dbscan', 'hdbscan', 'agglomerative']),
              help='Clustering-Algorithmus')
@click.option('--dr', '-d', default='pca',
              type=click.Choice(['pca', 'umap', 'tsne']),
              help='Dimensionsreduktions-Methode')
@click.option('--k', default=None, type=int, help='Anzahl Cluster (für kmeans/agglomerative)')
@click.option('--eps', default=None, type=float, help='Eps-Parameter für DBSCAN')
@click.option('--min-samples', default=None, type=int, help='Min-Samples für DBSCAN/HDBSCAN')
@click.option('--n-components', default=2, type=int, help='Anzahl Komponenten für DR')
@click.option('--seed', default=42, type=int, help='Random Seed')
@click.option('--config', '-c', help='Konfigurationsdatei (YAML/JSON)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose Ausgabe')
def cluster(input, output, algo, dr, k, eps, min_samples, n_components, seed, config, verbose):
    """Führt Clustering auf einer Datei durch"""
    
    if verbose:
        click.echo(f" Starte Clustering mit {algo} und {dr}")
        click.echo(f" Eingabe: {input}")
        click.echo(f" Output: {output}")
    
    try:
        # Seed setzen
        seed_manager = SeedManager()
        seed_manager.set_seed(seed)
        
        # Konfiguration laden oder erstellen
        if config:
            pipeline_config = load_config_file(config)
        else:
            pipeline_config = create_default_config(algo, dr, k, eps, min_samples, n_components, seed)
        
        # Daten laden
        if verbose:
            click.echo(" Lade Daten...")
        
        data_processor = DataProcessor(random_state=seed)
        
        # Dateityp erkennen
        file_extension = Path(input).suffix.lower()
        if file_extension == '.csv':
            df = data_processor.load_data(input, 'csv')
        elif file_extension == '.parquet':
            df = data_processor.load_data(input, 'parquet')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        if verbose:
            click.echo(f" Daten geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
        
        # Spalten analysieren
        column_analysis = data_processor.analyze_columns(df)
        
        # Numerische Spalten auswählen
        numeric_columns = column_analysis['numeric_columns']
        if not numeric_columns:
            click.echo("❌ Keine numerischen Spalten gefunden!")
            sys.exit(1)
        
        if verbose:
            click.echo(f" Verwende {len(numeric_columns)} numerische Spalten")
        
        # Preprocessing
        if verbose:
            click.echo(" Führe Preprocessing durch...")
        
        preprocessing_config = pipeline_config['preprocessing']
        df_processed = data_processor.preprocess_data(
            df,
            selected_columns=numeric_columns,
            **preprocessing_config
        )
        
        if verbose:
            click.echo(f" Preprocessing abgeschlossen: {df_processed.shape}")
        
        # Dimensionsreduktion
        if verbose:
            click.echo(f" Führe {dr} durch...")
        
        dr_reducer = DimensionalityReducer(random_state=seed)
        dr_config = pipeline_config['dimensionality_reduction']
        
        X_reduced = dr_reducer.fit_transform(
            df_processed.values,
            method=dr,
            **dr_config['parameters']
        )
        
        if verbose:
            click.echo(f" Dimensionsreduktion abgeschlossen: {X_reduced.shape}")
        
        # Clustering
        if verbose:
            click.echo(f" Führe {algo} Clustering durch...")
        
        clusterer = Clusterer(random_state=seed)
        cluster_config = pipeline_config['clustering']
        
        labels = clusterer.fit_predict(
            X_reduced,
            method=algo,
            **cluster_config['parameters']
        )
        
        n_clusters = len(np.unique(labels))
        outlier_count = np.sum(labels == -1)
        
        if verbose:
            click.echo(f" Clustering abgeschlossen: {n_clusters} Cluster, {outlier_count} Outlier")
        
        # Metriken berechnen
        if verbose:
            click.echo(" Berechne Metriken...")
        
        metrics_calculator = ClusteringMetrics()
        metrics = metrics_calculator.compute_all_metrics(X_reduced, labels)
        
        if verbose:
            if metrics.get('silhouette_score'):
                click.echo(f" Silhouette Score: {metrics['silhouette_score']:.3f}")
            if metrics.get('davies_bouldin_score'):
                click.echo(f" Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
        
        # Export
        if verbose:
            click.echo(" Exportiere Ergebnisse...")
        
        exporter = DataExporter(output)
        exported_files = exporter.export_complete_pipeline(
            df,
            labels,
            pipeline_config,
            metrics,
            dr_reducer,
            clusterer
        )
        
        if verbose:
            click.echo("Export abgeschlossen:")
            for file_type, filepath in exported_files.items():
                click.echo(f"  {file_type}: {filepath}")
        
        click.echo("Clustering erfolgreich abgeschlossen!")
        
    except Exception as e:
        click.echo(f"Fehler: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option('--output', '-o', default='sample_data.csv', help='Output-Datei')
@click.option('--n-samples', default=1000, type=int, help='Anzahl Samples')
@click.option('--n-features', default=10, type=int, help='Anzahl Features')
@click.option('--n-clusters', default=3, type=int, help='Anzahl Cluster')
@click.option('--format', default='csv', type=click.Choice(['csv', 'parquet']), help='Output-Format')
def generate_sample(output, n_samples, n_features, n_clusters, format):
    """Generiert Beispieldaten"""
    
    click.echo("Generiere Beispieldaten...")
    click.echo(f"  Samples: {n_samples}")
    click.echo(f"  Features: {n_features}")
    click.echo(f"  Cluster: {n_clusters}")
    
    try:
        df = create_sample_data(n_samples, n_features, n_clusters)
        
        if format == 'csv':
            df.to_csv(output, index=False)
        elif format == 'parquet':
            df.to_parquet(output, index=False)
        
        click.echo(f"Beispieldaten erstellt: {output}")
        click.echo(f"Shape: {df.shape}")
        
    except Exception as e:
        click.echo(f"Fehler: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option('--input', '-i', required=True, help='Eingabe-Datei')
@click.option('--output', '-o', default='config.yaml', help='Output-Konfiguration')
@click.option('--algo', '-a', default='kmeans',
              type=click.Choice(['kmeans', 'dbscan', 'hdbscan', 'agglomerative']),
              help='Clustering-Algorithmus')
@click.option('--dr', '-d', default='pca',
              type=click.Choice(['pca', 'umap', 'tsne']),
              help='Dimensionsreduktions-Methode')
def suggest_config(input, output, algo, dr):
    """Schlägt optimale Konfiguration basierend auf Daten vor"""
    
    click.echo(f" Analysiere Daten: {input}")
    
    try:
        # Daten laden
        data_processor = DataProcessor()
        
        file_extension = Path(input).suffix.lower()
        if file_extension == '.csv':
            df = data_processor.load_data(input, 'csv')
        elif file_extension == '.parquet':
            df = data_processor.load_data(input, 'parquet')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Spalten analysieren
        column_analysis = data_processor.analyze_columns(df)
        numeric_columns = column_analysis['numeric_columns']
        
        if not numeric_columns:
            click.echo("❌ Keine numerischen Spalten gefunden!")
            sys.exit(1)
        
        # Preprocessing
        df_processed = data_processor.preprocess_data(df, selected_columns=numeric_columns)
        
        n_samples, n_features = df_processed.shape
        
        click.echo(f" Daten-Charakteristika:")
        click.echo(f"  Samples: {n_samples}")
        click.echo(f"  Features: {n_features}")
        
        # Optimale Parameter vorschlagen
        dr_params = get_dr_optimal_params(dr, n_samples, n_features)
        cluster_params = get_cluster_optimal_params(algo, n_samples, n_features)
        
        # Konfiguration erstellen
        config = create_pipeline_config(
            preprocessing={
                'handle_missing': 'impute',
                'imputation_method': 'mean',
                'scaling_method': 'standard',
                'one_hot_encode': True,
                'max_categories': 50,
                'remove_duplicates': True,
                'variance_threshold': 0.0
            },
            dr_method=dr,
            dr_params=dr_params,
            cluster_method=algo,
            cluster_params=cluster_params,
            random_seed=42
        )
        
        # Konfiguration speichern (numpy types konvertieren)
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
        
        with open(output, 'w', encoding='utf-8') as f:
            yaml.dump(config_serializable, f, default_flow_style=False, allow_unicode=True)
        
        click.echo(f" Konfiguration erstellt: {output}")
        click.echo(f" Empfohlene Parameter:")
        click.echo(f"  DR ({dr}): {dr_params}")
        click.echo(f"  Clustering ({algo}): {cluster_params}")
        
    except Exception as e:
        click.echo(f"Fehler: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option('--input', '-i', required=True, help='Eingabe-Datei')
@click.option('--algo', '-a', default='kmeans',
              type=click.Choice(['kmeans', 'dbscan', 'hdbscan', 'agglomerative']),
              help='Clustering-Algorithmus')
@click.option('--k-range', default='2,15', help='K-Bereich für Optimierung (min,max)')
@click.option('--seed', default=42, type=int, help='Random Seed')
def optimize_k(input, algo, k_range, seed):
    """Findet optimale Anzahl von Clustern"""
    
    if algo not in ['kmeans', 'agglomerative']:
        click.echo(f"❌ K-Optimierung nur für {algo} unterstützt!")
        sys.exit(1)
    
    click.echo(f" Finde optimale K für {algo}...")
    
    try:
        # Seed setzen
        seed_manager = SeedManager()
        seed_manager.set_seed(seed)
        
        # Daten laden und vorbereiten
        data_processor = DataProcessor(random_state=seed)
        
        file_extension = Path(input).suffix.lower()
        if file_extension == '.csv':
            df = data_processor.load_data(input, 'csv')
        elif file_extension == '.parquet':
            df = data_processor.load_data(input, 'parquet')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Preprocessing
        column_analysis = data_processor.analyze_columns(df)
        numeric_columns = column_analysis['numeric_columns']
        df_processed = data_processor.preprocess_data(df, selected_columns=numeric_columns)
        
        # K-Bereich parsen
        k_min, k_max = map(int, k_range.split(','))
        
        # Optimale K finden
        from cluster import find_optimal_k
        
        result = find_optimal_k(
            df_processed.values,
            k_range=(k_min, k_max),
            method=algo,
            random_state=seed
        )
        
        click.echo(f" Optimale K: {result['best_k']}")
        click.echo(f" Bester Score: {result['best_score']:.3f}")
        click.echo(f" Alle Scores:")
        
        for k, score in zip(result['k_values'], result['scores']):
            marker = "👑" if k == result['best_k'] else "  "
            click.echo(f"  {marker} k={k}: {score:.3f}")
        
    except Exception as e:
        click.echo(f"Fehler: {str(e)}", err=True)
        sys.exit(1)


def load_config_file(filepath: str) -> Dict[str, Any]:
    """Lädt eine Konfigurationsdatei"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    if filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {filepath}")


def create_default_config(
    algo: str,
    dr: str,
    k: Optional[int],
    eps: Optional[float],
    min_samples: Optional[int],
    n_components: int,
    seed: int
) -> Dict[str, Any]:
    """Erstellt eine Standard-Konfiguration"""
    
    # Preprocessing-Konfiguration
    preprocessing = {
        'handle_missing': 'impute',
        'imputation_method': 'mean',
        'scaling_method': 'standard',
        'one_hot_encode': True,
        'max_categories': 50,
        'remove_duplicates': True,
        'variance_threshold': 0.0
    }
    
    # Dimensionsreduktions-Konfiguration
    dr_params = {'n_components': n_components}
    if dr == 'umap':
        dr_params.update({'n_neighbors': 15, 'min_dist': 0.1})
    elif dr == 'tsne':
        dr_params.update({'perplexity': 30, 'learning_rate': 200})
    
    # Clustering-Konfiguration
    cluster_params = {}
    if algo == 'kmeans':
        cluster_params['n_clusters'] = k or 8
    elif algo == 'dbscan':
        cluster_params['eps'] = eps or 0.5
        cluster_params['min_samples'] = min_samples or 5
    elif algo == 'hdbscan':
        cluster_params['min_cluster_size'] = min_samples or 5
    elif algo == 'agglomerative':
        cluster_params['n_clusters'] = k or 8
        cluster_params['linkage'] = 'ward'
    
    return create_pipeline_config(
        preprocessing=preprocessing,
        dr_method=dr,
        dr_params=dr_params,
        cluster_method=algo,
        cluster_params=cluster_params,
        random_seed=seed
    )


if __name__ == '__main__':
    main()
