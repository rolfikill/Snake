"""
Streamlit UI für die Data Clustering App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
import io
import os

# Import der App-Module
from data import DataProcessor, create_sample_data
from dr import DimensionalityReducer, get_method_info as get_dr_info, recommend_method as recommend_dr_method
from cluster import Clusterer, get_method_info as get_cluster_info, recommend_method as recommend_cluster_method, find_optimal_k
from metrics import ClusteringMetrics, create_metrics_dataframe
from viz import ClusteringVisualizer
from state import SessionState, SeedManager, get_step_info, can_proceed_to_step
from io_utils import DataExporter, create_streamlit_download_button, validate_export_data


def main():
    """Hauptfunktion der Streamlit App"""
    
    # Page Config
    st.set_page_config(
        page_title="Data Clustering App",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS für bessere Darstellung
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2ca02c;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Data Clustering App</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Lokale App für ungelabelte Datensätze</strong><br>
    Upload → Vorverarbeitung → Dimensionsreduktion → Clustering → Visualisierung
    </div>
    """, unsafe_allow_html=True)
    
    # SessionState initialisieren
    session_state = SessionState()
    seed_manager = SeedManager()
    
    # Sidebar für Navigation
    render_sidebar(session_state)
    
    # Hauptinhalt basierend auf aktuellem Schritt
    current_step = session_state.get_current_step()
    
    if current_step == 'upload':
        render_upload_step(session_state, seed_manager)
    elif current_step == 'preprocessing':
        render_preprocessing_step(session_state)
    elif current_step == 'dimensionality_reduction':
        render_dr_step(session_state)
    elif current_step == 'clustering':
        render_clustering_step(session_state)
    elif current_step == 'results':
        render_results_step(session_state)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
    Data Clustering App v0.1.0 | Lokal & Offline | Keine Telemetrie
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(session_state: SessionState):
    """Rendert die Sidebar mit Navigation"""
    
    st.sidebar.title("📋 Navigation")
    
    # Schritt-Informationen
    step_info = get_step_info()
    steps = ['upload', 'preprocessing', 'dimensionality_reduction', 'clustering', 'results']
    
    current_step = session_state.get_current_step()
    
    for step in steps:
        step_data = step_info[step]
        can_proceed = can_proceed_to_step(step, session_state)
        
        if step == current_step:
            st.sidebar.markdown(f"**{step_data['icon']} {step_data['name']}** (Aktuell)")
        elif can_proceed:
            if st.sidebar.button(f"{step_data['icon']} {step_data['name']}", key=f"nav_{step}"):
                session_state.set_current_step(step)
                st.rerun()
        else:
            st.sidebar.markdown(f"🔒 {step_data['name']} (Nicht verfügbar)")
        
        st.sidebar.markdown(f"<small>{step_data['description']}</small>", unsafe_allow_html=True)
        st.sidebar.markdown("---")
    
    # Status-Übersicht
    st.sidebar.markdown("### 📊 Status")
    status = session_state.get_state_summary()
    
    st.sidebar.markdown(f"**Daten geladen:** {'✅' if status['data_loaded'] else '❌'}")
    if status['data_shape']:
        st.sidebar.markdown(f"Shape: {status['data_shape'][0]} × {status['data_shape'][1]}")
    
    st.sidebar.markdown(f"**Verarbeitet:** {'✅' if status['processed'] else '❌'}")
    if status['processed_shape']:
        st.sidebar.markdown(f"Shape: {status['processed_shape'][0]} × {status['processed_shape'][1]}")
    
    st.sidebar.markdown(f"**DR durchgeführt:** {'✅' if status['dr_computed'] else '❌'}")
    if status['dr_shape']:
        st.sidebar.markdown(f"Shape: {status['dr_shape'][0]} × {status['dr_shape'][1]}")
    
    st.sidebar.markdown(f"**Clustering:** {'✅' if status['clustered'] else '❌'}")
    if status['n_clusters']:
        st.sidebar.markdown(f"Cluster: {status['n_clusters']}")
    
    # Random Seed
    st.sidebar.markdown("### 🎲 Random Seed")
    current_seed = session_state.get_random_seed()
    new_seed = st.sidebar.number_input("Seed", value=current_seed, min_value=0, max_value=2**32-1)
    if new_seed != current_seed:
        session_state.set_random_seed(new_seed)
        st.rerun()
    
    # Reset Button
    if st.sidebar.button("🔄 Alles zurücksetzen", type="secondary"):
        session_state.reset_state()
        st.rerun()


def render_upload_step(session_state: SessionState, seed_manager: SeedManager):
    """Rendert den Upload-Schritt"""
    
    st.markdown('<h2 class="step-header">📁 Daten-Upload</h2>', unsafe_allow_html=True)
    
    # Upload-Optionen
    upload_method = st.radio(
        "Upload-Methode wählen:",
        ["Datei hochladen", "Beispieldaten laden"],
        horizontal=True
    )
    
    if upload_method == "Datei hochladen":
        uploaded_file = st.file_uploader(
            "CSV oder Parquet Datei hochladen",
            type=['csv', 'parquet'],
            help="Unterstützte Formate: CSV, Parquet"
        )
        
        if uploaded_file is not None:
            try:
                # Datei-Informationen
                file_info = {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'type': uploaded_file.type
                }
                
                # Daten laden
                data_processor = DataProcessor(random_state=session_state.get_random_seed())
                df = data_processor.load_data_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
                
                # In SessionState speichern
                session_state.set_data(df, file_info)
                
                st.success(f"✅ Datei erfolgreich geladen: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"❌ Fehler beim Laden der Datei: {str(e)}")
    
    else:  # Beispieldaten
        st.markdown("### 🎲 Beispieldaten generieren")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_samples = st.number_input("Anzahl Samples", min_value=100, max_value=10000, value=1000)
        with col2:
            n_features = st.number_input("Anzahl Features", min_value=2, max_value=50, value=10)
        with col3:
            n_clusters = st.number_input("Anzahl Cluster", min_value=2, max_value=10, value=3)
        
        if st.button("🎲 Beispieldaten generieren"):
            try:
                df = create_sample_data(n_samples, n_features, n_clusters)
                session_state.set_data(df, {'name': 'sample_data.csv', 'size': 0, 'type': 'sample'})
                st.success(f"✅ Beispieldaten generiert: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
            except Exception as e:
                st.error(f"❌ Fehler beim Generieren: {str(e)}")
    
    # Datenvorschau
    if session_state.is_data_loaded():
        df = session_state.get_data()
        
        st.markdown("### 📊 Datenvorschau")
        
        # Grundlegende Informationen
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Zeilen", df.shape[0])
        with col2:
            st.metric("Spalten", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplikate", df.duplicated().sum())
        
        # Spalten-Typen
        st.markdown("#### Spalten-Informationen")
        column_info = pd.DataFrame({
            'Spalte': df.columns,
            'Typ': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Null': df.isnull().sum(),
            'Eindeutig': df.nunique()
        })
        st.dataframe(column_info, use_container_width=True)
        
        # Datenvorschau
        st.markdown("#### Erste 10 Zeilen")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Weiter-Button
        if st.button("➡️ Weiter zur Vorverarbeitung", type="primary"):
            session_state.set_current_step('preprocessing')
            st.rerun()


def render_preprocessing_step(session_state: SessionState):
    """Rendert den Preprocessing-Schritt"""
    
    st.markdown('<h2 class="step-header">🔧 Vorverarbeitung</h2>', unsafe_allow_html=True)
    
    if not session_state.is_data_loaded():
        st.warning("⚠️ Bitte zuerst Daten laden!")
        return
    
    df = session_state.get_data()
    data_processor = DataProcessor(random_state=session_state.get_random_seed())
    
    # Spalten-Analyse
    if session_state.get_column_analysis() is None:
        with st.spinner("Analysiere Spalten..."):
            column_analysis = data_processor.analyze_columns(df)
            session_state.set_column_analysis(column_analysis)
    
    column_analysis = session_state.get_column_analysis()
    
    # Spalten-Auswahl
    st.markdown("### 📋 Spalten auswählen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Numerische Spalten")
        numeric_columns = st.multiselect(
            "Numerische Spalten auswählen:",
            options=column_analysis['numeric_columns'],
            default=column_analysis['numeric_columns']
        )
    
    with col2:
        st.markdown("#### Kategorische Spalten")
        categorical_columns = st.multiselect(
            "Kategorische Spalten (für One-Hot Encoding):",
            options=column_analysis['categorical_columns'],
            default=[]
        )
    
    # Preprocessing-Optionen
    st.markdown("### ⚙️ Preprocessing-Optionen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        handle_missing = st.selectbox(
            "Missing Values behandeln:",
            ["drop", "impute"],
            help="'drop': Zeilen mit Missing Values entfernen, 'impute': Werte ersetzen"
        )
        
        if handle_missing == "impute":
            imputation_method = st.selectbox(
                "Imputation-Methode:",
                ["mean", "median", "most_frequent"],
                help="Methode zum Ersetzen von Missing Values"
            )
        else:
            imputation_method = "mean"
        
        scaling_method = st.selectbox(
            "Skalierung:",
            ["none", "standard", "minmax", "robust"],
            help="Skalierungsmethode für numerische Features"
        )
    
    with col2:
        one_hot_encode = st.checkbox(
            "One-Hot Encoding für kategorische Spalten",
            value=True,
            help="Kategorische Spalten in binäre Features umwandeln"
        )
        
        if one_hot_encode:
            max_categories = st.number_input(
                "Max. Kategorien pro Spalte:",
                min_value=2,
                max_value=100,
                value=50,
                help="Spalten mit mehr Kategorien werden ignoriert"
            )
        else:
            max_categories = 50
        
        remove_duplicates = st.checkbox(
            "Duplikate entfernen",
            value=True
        )
        
        variance_threshold = st.slider(
            "Varianz-Schwellwert:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help="Features mit geringerer Varianz entfernen"
        )
    
    # Sampling für große Datensätze
    if len(df) > 20000:
        st.markdown("### 📊 Sampling für große Datensätze")
        st.warning("⚠️ Datensatz ist groß (>20k Zeilen). Sampling empfohlen für bessere Performance.")
        
        sample_size = st.slider(
            "Stichprobengröße (%):",
            min_value=5,
            max_value=100,
            value=min(50, 100),
            help="Prozentsatz der Daten für die Analyse verwenden"
        )
    else:
        sample_size = 100
    
    # Preprocessing durchführen
    if st.button("🔧 Preprocessing durchführen", type="primary"):
        try:
            with st.spinner("Führe Preprocessing durch..."):
                # Alle ausgewählten Spalten
                selected_columns = numeric_columns + categorical_columns
                
                if not selected_columns:
                    st.error("❌ Bitte mindestens eine Spalte auswählen!")
                    return
                
                # Preprocessing
                df_processed = data_processor.preprocess_data(
                    df,
                    selected_columns=selected_columns,
                    handle_missing=handle_missing,
                    imputation_method=imputation_method,
                    scaling_method=scaling_method,
                    one_hot_encode=one_hot_encode,
                    max_categories=max_categories,
                    remove_duplicates=remove_duplicates,
                    variance_threshold=variance_threshold
                )
                
                # Sampling
                if sample_size < 100:
                    df_processed = data_processor.sample_data(df_processed, sample_size / 100)
                
                # In SessionState speichern
                session_state.set_processed_data(df_processed)
                
                st.success(f"✅ Preprocessing abgeschlossen!")
                st.info(f"📊 Verarbeitete Daten: {df_processed.shape[0]} Zeilen, {df_processed.shape[1]} Features")
                
                # Konfiguration speichern
                config = {
                    'selected_columns': selected_columns,
                    'handle_missing': handle_missing,
                    'imputation_method': imputation_method,
                    'scaling_method': scaling_method,
                    'one_hot_encode': one_hot_encode,
                    'max_categories': max_categories,
                    'remove_duplicates': remove_duplicates,
                    'variance_threshold': variance_threshold,
                    'sample_size': sample_size
                }
                session_state.set_pipeline_config(config)
                
        except Exception as e:
            st.error(f"❌ Fehler beim Preprocessing: {str(e)}")
    
    # Weiter-Button
    if session_state.is_processed():
        if st.button("➡️ Weiter zur Dimensionsreduktion", type="primary"):
            session_state.set_current_step('dimensionality_reduction')
            st.rerun()


def render_dr_step(session_state: SessionState):
    """Rendert den Dimensionsreduktions-Schritt"""
    
    st.markdown('<h2 class="step-header">📉 Dimensionsreduktion</h2>', unsafe_allow_html=True)
    
    if not session_state.is_processed():
        st.warning("⚠️ Bitte zuerst Preprocessing durchführen!")
        return
    
    df_processed = session_state.get_processed_data()
    
    # Methode auswählen
    st.markdown("### 🎯 Methode auswählen")
    
    dr_method = st.selectbox(
        "Dimensionsreduktions-Methode:",
        ["pca", "umap", "tsne"],
        format_func=lambda x: {
            "pca": "PCA - Principal Component Analysis",
            "umap": "UMAP - Uniform Manifold Approximation",
            "tsne": "t-SNE - t-Distributed Stochastic Neighbor Embedding"
        }[x]
    )
    
    # Methode-Informationen anzeigen
    dr_info = get_dr_info()
    if dr_method in dr_info:
        info = dr_info[dr_method]
        with st.expander(f"ℹ️ Informationen zu {info['name']}"):
            st.markdown(f"**Beschreibung:** {info['description']}")
            st.markdown("**Vorteile:**")
            for pro in info['pros']:
                st.markdown(f"• {pro}")
            st.markdown("**Nachteile:**")
            for con in info['cons']:
                st.markdown(f"• {con}")
            st.markdown(f"**Am besten für:** {info['best_for']}")
    
    # Parameter
    st.markdown("### ⚙️ Parameter")
    
    n_components = st.number_input(
        "Anzahl Komponenten:",
        min_value=2,
        max_value=min(50, df_processed.shape[1]),
        value=2,
        help="Anzahl der Ausgabe-Dimensionen"
    )
    
    # Spezifische Parameter
    if dr_method == "pca":
        whiten = st.checkbox("Whiten", value=False, help="Normalisierung der Komponenten")
        svd_solver = st.selectbox("SVD Solver", ["auto", "full", "arpack", "randomized"])
        
        dr_params = {
            'n_components': n_components,
            'whiten': whiten,
            'svd_solver': svd_solver
        }
    
    elif dr_method == "umap":
        col1, col2 = st.columns(2)
        with col1:
            n_neighbors = st.number_input(
                "n_neighbors:",
                min_value=2,
                max_value=min(100, len(df_processed) - 1),
                value=min(15, len(df_processed) - 1),
                help="Anzahl der Nachbarn für lokale Struktur"
            )
        with col2:
            min_dist = st.slider(
                "min_dist:",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Minimale Distanz zwischen Punkten"
            )
        
        metric = st.selectbox("Metrik", ["euclidean", "manhattan", "chebyshev", "minkowski"])
        
        dr_params = {
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric
        }
    
    elif dr_method == "tsne":
        col1, col2 = st.columns(2)
        with col1:
            perplexity = st.number_input(
                "Perplexity:",
                min_value=5.0,
                max_value=min(50.0, len(df_processed) / 3),
                value=min(30.0, len(df_processed) / 3),
                help="Anzahl der effektiven Nachbarn"
            )
        with col2:
            learning_rate = st.number_input(
                "Learning Rate:",
                min_value=10.0,
                max_value=1000.0,
                value=200.0,
                help="Lernrate für die Optimierung"
            )
        
        n_iter = st.number_input(
            "Iterationen:",
            min_value=250,
            max_value=5000,
            value=1000,
            help="Anzahl der Optimierungs-Iterationen"
        )
        
        dr_params = {
            'n_components': n_components,
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'n_iter': n_iter
        }
    
    # Dimensionsreduktion durchführen
    if st.button("📉 Dimensionsreduktion durchführen", type="primary"):
        try:
            with st.spinner("Führe Dimensionsreduktion durch..."):
                dr_reducer = DimensionalityReducer(random_state=session_state.get_random_seed())
                X_reduced = dr_reducer.fit_transform(
                    df_processed.values,
                    method=dr_method,
                    **dr_params
                )
                
                # In SessionState speichern
                session_state.set_dr_model(dr_reducer, X_reduced)
                
                st.success(f"✅ Dimensionsreduktion abgeschlossen!")
                st.info(f"📊 Reduzierte Daten: {X_reduced.shape[0]} Punkte, {X_reduced.shape[1]} Dimensionen")
                
                # Modell-Informationen
                model_info = dr_reducer.get_model_info()
                if dr_method == "pca" and model_info.get('total_variance_explained'):
                    st.info(f"📈 Erklärte Varianz: {model_info['total_variance_explained']:.1%}")
                
                # Konfiguration aktualisieren
                config = session_state.get_pipeline_config()
                config['dr_method'] = dr_method
                config['dr_params'] = dr_params
                session_state.set_pipeline_config(config)
                
        except Exception as e:
            st.error(f"❌ Fehler bei der Dimensionsreduktion: {str(e)}")
    
    # Weiter-Button
    if session_state.is_dr_computed():
        if st.button("➡️ Weiter zum Clustering", type="primary"):
            session_state.set_current_step('clustering')
            st.rerun()


def render_clustering_step(session_state: SessionState):
    """Rendert den Clustering-Schritt"""
    
    st.markdown('<h2 class="step-header">🎯 Clustering</h2>', unsafe_allow_html=True)
    
    if not session_state.is_dr_computed():
        st.warning("⚠️ Bitte zuerst Dimensionsreduktion durchführen!")
        return
    
    X_reduced = session_state.get_dr_data()
    
    # Methode auswählen
    st.markdown("### 🎯 Algorithmus auswählen")
    
    # Verfügbare Algorithmen basierend auf Installation
    available_algorithms = ["kmeans", "dbscan", "agglomerative"]
    algorithm_names = {
        "kmeans": "K-Means",
        "dbscan": "DBSCAN",
        "agglomerative": "Agglomerative Clustering"
    }
    
    # HDBSCAN hinzufügen falls verfügbar
    try:
        import hdbscan
        available_algorithms.append("hdbscan")
        algorithm_names["hdbscan"] = "HDBSCAN"
    except ImportError:
        pass
    
    cluster_method = st.selectbox(
        "Clustering-Algorithmus:",
        available_algorithms,
        format_func=lambda x: algorithm_names[x]
    )
    
    # Methode-Informationen anzeigen
    cluster_info = get_cluster_info()
    if cluster_method in cluster_info:
        info = cluster_info[cluster_method]
        with st.expander(f"ℹ️ Informationen zu {info['name']}"):
            st.markdown(f"**Beschreibung:** {info['description']}")
            st.markdown("**Vorteile:**")
            for pro in info['pros']:
                st.markdown(f"• {pro}")
            st.markdown("**Nachteile:**")
            for con in info['cons']:
                st.markdown(f"• {con}")
            st.markdown(f"**Am besten für:** {info['best_for']}")
    
    # Parameter
    st.markdown("### ⚙️ Parameter")
    
    if cluster_method == "kmeans":
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.number_input(
                "Anzahl Cluster:",
                min_value=2,
                max_value=20,
                value=8,
                help="Anzahl der zu findenden Cluster"
            )
        with col2:
            n_init = st.selectbox(
                "n_init:",
                ["auto", 10, 20, 50],
                help="Anzahl der Initialisierungen"
            )
        
        max_iter = st.number_input(
            "Max. Iterationen:",
            min_value=100,
            max_value=1000,
            value=300
        )
        
        cluster_params = {
            'n_clusters': n_clusters,
            'n_init': n_init,
            'max_iter': max_iter
        }
        
        # K-Optimierung
        if st.checkbox("🔍 Optimale K finden (Elbow-Methode)"):
            k_range = st.slider(
                "K-Bereich:",
                min_value=2,
                max_value=15,
                value=(2, 10)
            )
            
            if st.button("🎯 Optimale K finden"):
                try:
                    with st.spinner("Finde optimale K..."):
                        result = find_optimal_k(
                            X_reduced,
                            k_range=k_range,
                            method='kmeans',
                            random_state=session_state.get_random_seed()
                        )
                        
                        st.success(f"✅ Optimale K: {result['best_k']} (Score: {result['best_score']:.3f})")
                        
                        # Plot der Scores
                        import plotly.graph_objects as go
                        fig = go.Figure(data=[
                            go.Scatter(
                                x=result['k_values'],
                                y=result['scores'],
                                mode='lines+markers',
                                name='Silhouette Score'
                            )
                        ])
                        fig.update_layout(
                            title="Elbow-Methode - Silhouette Scores",
                            xaxis_title="Anzahl Cluster (k)",
                            yaxis_title="Silhouette Score"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Empfohlene K setzen
                        if st.button(f"✅ K = {result['best_k']} verwenden"):
                            cluster_params['n_clusters'] = result['best_k']
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"❌ Fehler bei K-Optimierung: {str(e)}")
    
    elif cluster_method == "dbscan":
        col1, col2 = st.columns(2)
        with col1:
            eps = st.number_input(
                "eps:",
                min_value=0.01,
                max_value=10.0,
                value=0.5,
                step=0.01,
                help="Maximale Distanz zwischen Nachbarn"
            )
        with col2:
            min_samples = st.number_input(
                "min_samples:",
                min_value=2,
                max_value=50,
                value=5,
                help="Mindestanzahl Samples pro Cluster"
            )
        
        metric = st.selectbox("Metrik", ["euclidean", "manhattan", "chebyshev", "minkowski"])
        
        cluster_params = {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric
        }
        
        # eps-Schätzung
        if st.checkbox("🔍 eps automatisch schätzen"):
            if st.button("📏 eps schätzen"):
                try:
                    from .cluster import estimate_eps
                    estimated_eps = estimate_eps(X_reduced, min_samples)
                    st.info(f"📏 Geschätztes eps: {estimated_eps:.3f}")
                    if st.button(f"✅ eps = {estimated_eps:.3f} verwenden"):
                        cluster_params['eps'] = estimated_eps
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Fehler bei eps-Schätzung: {str(e)}")
    
    elif cluster_method == "hdbscan":
        col1, col2 = st.columns(2)
        with col1:
            min_cluster_size = st.number_input(
                "min_cluster_size:",
                min_value=2,
                max_value=50,
                value=5,
                help="Mindestgröße eines Clusters"
            )
        with col2:
            min_samples = st.number_input(
                "min_samples:",
                min_value=1,
                max_value=50,
                value=None,
                help="Mindestanzahl Samples (None = min_cluster_size)"
            )
        
        metric = st.selectbox("Metrik", ["euclidean", "manhattan", "chebyshev", "minkowski"])
        
        cluster_params = {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'metric': metric
        }
    
    elif cluster_method == "agglomerative":
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.number_input(
                "Anzahl Cluster:",
                min_value=2,
                max_value=20,
                value=8
            )
        with col2:
            linkage = st.selectbox(
                "Linkage:",
                ["ward", "complete", "average", "single"],
                help="Verknüpfungs-Methode"
            )
        
        metric = st.selectbox("Metrik", ["euclidean", "manhattan", "chebyshev", "minkowski"])
        
        cluster_params = {
            'n_clusters': n_clusters,
            'linkage': linkage,
            'metric': metric
        }
    
    # Clustering durchführen
    if st.button("🎯 Clustering durchführen", type="primary"):
        try:
            with st.spinner("Führe Clustering durch..."):
                clusterer = Clusterer(random_state=session_state.get_random_seed())
                labels = clusterer.fit_predict(
                    X_reduced,
                    method=cluster_method,
                    **cluster_params
                )
                
                # In SessionState speichern
                session_state.set_cluster_model(clusterer, labels)
                
                # Cluster-Informationen
                cluster_info = clusterer.get_cluster_info()
                n_clusters = cluster_info['n_clusters']
                outlier_count = cluster_info['outlier_count']
                
                st.success(f"✅ Clustering abgeschlossen!")
                st.info(f"🎯 Gefunden: {n_clusters} Cluster, {outlier_count} Outlier")
                
                # Metriken berechnen
                with st.spinner("Berechne Metriken..."):
                    metrics_calculator = ClusteringMetrics()
                    metrics = metrics_calculator.compute_all_metrics(X_reduced, labels)
                    session_state.set_metrics(metrics)
                    
                    # Wichtige Metriken anzeigen
                    if metrics.get('silhouette_score'):
                        st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
                    if metrics.get('davies_bouldin_score'):
                        st.metric("Davies-Bouldin Score", f"{metrics['davies_bouldin_score']:.3f}")
                
                # Konfiguration aktualisieren
                config = session_state.get_pipeline_config()
                config['cluster_method'] = cluster_method
                config['cluster_params'] = cluster_params
                session_state.set_pipeline_config(config)
                
        except Exception as e:
            st.error(f"❌ Fehler beim Clustering: {str(e)}")
    
    # Weiter-Button
    if session_state.is_clustered():
        if st.button("➡️ Weiter zu den Ergebnissen", type="primary"):
            session_state.set_current_step('results')
            st.rerun()


def render_results_step(session_state: SessionState):
    """Rendert den Results-Schritt"""
    
    st.markdown('<h2 class="step-header">📊 Ergebnisse</h2>', unsafe_allow_html=True)
    
    if not session_state.is_clustered():
        st.warning("⚠️ Bitte zuerst Clustering durchführen!")
        return
    
    # Daten abrufen
    X_reduced = session_state.get_dr_data()
    labels = session_state.get_cluster_labels()
    metrics = session_state.get_metrics()
    df_original = session_state.get_data()
    
    # Visualisierung
    st.markdown("### 📈 Visualisierung")
    
    visualizer = ClusteringVisualizer()
    
    # Hover-Daten vorbereiten
    hover_columns = []
    if len(df_original.columns) <= 5:
        hover_columns = df_original.columns.tolist()
    else:
        # Erste 5 Spalten auswählen
        hover_columns = df_original.columns[:5].tolist()
    
    # Scatter Plot
    fig_scatter = visualizer.create_scatter_plot(
        X_reduced,
        labels,
        title="Clustering Visualisierung",
        hover_data=df_original,
        hover_columns=hover_columns
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Zusätzliche Plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster-Größen
        fig_sizes = visualizer.create_cluster_size_chart(labels)
        st.plotly_chart(fig_sizes, use_container_width=True)
    
    with col2:
        # Metriken
        fig_metrics = visualizer.create_metrics_chart(metrics)
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Silhouette-Analyse
    if metrics.get('silhouette_score') is not None:
        st.markdown("### 🔍 Silhouette-Analyse")
        fig_silhouette = visualizer.create_silhouette_plot(X_reduced, labels)
        st.plotly_chart(fig_silhouette, use_container_width=True)
    
    # Metriken-Tabelle
    st.markdown("### 📊 Detaillierte Metriken")
    metrics_df = create_metrics_dataframe(metrics)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Cluster-Details
    st.markdown("### 🎯 Cluster-Details")
    
    # Cluster-Statistiken
    unique_labels = np.unique(labels)
    cluster_stats = []
    
    for label in unique_labels:
        if label == -1:
            name = "Outlier"
        else:
            name = f"Cluster {label}"
        
        mask = labels == label
        count = np.sum(mask)
        percentage = (count / len(labels)) * 100
        
        cluster_stats.append({
            'Cluster': name,
            'Anzahl': count,
            'Prozent': f"{percentage:.1f}%"
        })
    
    cluster_stats_df = pd.DataFrame(cluster_stats)
    st.dataframe(cluster_stats_df, use_container_width=True)
    
    # Export
    st.markdown("### 💾 Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Gelabelte Daten
        if st.button("📄 Gelabelte Daten herunterladen"):
            try:
                # Validierung
                is_valid, error_msg = validate_export_data(df_original, labels)
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                    return
                
                # DataFrame mit Labels erstellen
                labeled_data = df_original.copy()
                labeled_data['cluster'] = labels
                
                # Download
                csv_data = create_streamlit_download_button(
                    labeled_data, "labeled_data.csv", "text/csv"
                )
                st.download_button(
                    label="📥 CSV herunterladen",
                    data=csv_data,
                    file_name="labeled_data.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Fehler beim Export: {str(e)}")
    
    with col2:
        # Konfiguration
        if st.button("⚙️ Konfiguration herunterladen"):
            try:
                config = session_state.get_pipeline_config()
                config['metadata'] = {
                    'created_at': pd.Timestamp.now().isoformat(),
                    'app_version': '0.1.0'
                }
                
                yaml_data = create_streamlit_download_button(
                    config, "pipeline_config.yaml", "text/yaml"
                )
                st.download_button(
                    label="📥 YAML herunterladen",
                    data=yaml_data,
                    file_name="pipeline_config.yaml",
                    mime="text/yaml"
                )
                
            except Exception as e:
                st.error(f"❌ Fehler beim Export: {str(e)}")
    
    with col3:
        # Metriken
        if st.button("📊 Metriken herunterladen"):
            try:
                metrics_export = metrics.copy()
                metrics_export['metadata'] = {
                    'created_at': pd.Timestamp.now().isoformat(),
                    'app_version': '0.1.0'
                }
                
                json_data = create_streamlit_download_button(
                    metrics_export, "metrics.json", "application/json"
                )
                st.download_button(
                    label="📥 JSON herunterladen",
                    data=json_data,
                    file_name="metrics.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"❌ Fehler beim Export: {str(e)}")
    
    # Neuer Durchlauf
    st.markdown("### 🔄 Neuer Durchlauf")
    if st.button("🆕 Neuen Durchlauf starten", type="secondary"):
        session_state.reset_state()
        st.rerun()


if __name__ == "__main__":
    main()
