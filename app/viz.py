"""
Visualisierungs-Modul mit Plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List, Tuple
import colorsys


class ClusteringVisualizer:
    """Klasse für Clustering-Visualisierung"""
    
    def __init__(self):
        self.color_palette = None
        self.default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
    
    def create_scatter_plot(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        title: str = "Clustering Visualisierung",
        hover_data: Optional[pd.DataFrame] = None,
        hover_columns: Optional[List[str]] = None,
        show_outliers: bool = True,
        point_size: float = 6.0,
        opacity: float = 0.7
    ) -> go.Figure:
        """
        Erstellt einen 2D Scatter Plot der Cluster
        
        Args:
            X: 2D Daten (x, y Koordinaten)
            labels: Cluster-Labels
            title: Plot-Titel
            hover_data: DataFrame mit zusätzlichen Daten für Hover
            hover_columns: Spalten die im Hover angezeigt werden sollen
            show_outliers: Ob Outlier (Label -1) angezeigt werden sollen
            point_size: Größe der Punkte
            opacity: Transparenz der Punkte
        """
        if X.shape[1] != 2:
            raise ValueError("X muss 2D Daten enthalten (x, y Koordinaten)")
        
        # Eindeutige Labels finden
        unique_labels = np.unique(labels)
        
        # Outlier entfernen wenn gewünscht
        if not show_outliers:
            unique_labels = unique_labels[unique_labels != -1]
        
        # Farben generieren
        colors = self._generate_colors(len(unique_labels))
        
        fig = go.Figure()
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            
            # Hover-Text erstellen
            hover_text = self._create_hover_text(
                mask, hover_data, hover_columns, label
            )
            
            # Label-Name
            if label == -1:
                label_name = "Outlier"
                marker_symbol = 'x'
                marker_size = point_size * 1.5
            else:
                label_name = f"Cluster {label}"
                marker_symbol = 'circle'
                marker_size = point_size
            
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                name=label_name,
                marker=dict(
                    size=marker_size,
                    color=colors[i],
                    opacity=opacity,
                    symbol=marker_symbol,
                    line=dict(width=0.5, color='white')
                ),
                text=hover_text,
                hovertemplate='<b>%{text}</b><br>' +
                             'X: %{x:.3f}<br>' +
                             'Y: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))
        
        # Layout anpassen
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            width=800,
            height=600,
            margin=dict(l=50, r=150, t=50, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Grid hinzufügen
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_cluster_size_chart(
        self,
        labels: np.ndarray,
        title: str = "Cluster-Größen"
    ) -> go.Figure:
        """Erstellt ein Balkendiagramm der Cluster-Größen"""
        
        unique_labels = np.unique(labels)
        cluster_sizes = []
        cluster_names = []
        
        for label in unique_labels:
            size = np.sum(labels == label)
            cluster_sizes.append(size)
            if label == -1:
                cluster_names.append("Outlier")
            else:
                cluster_names.append(f"Cluster {label}")
        
        # Farben generieren
        colors = self._generate_colors(len(unique_labels))
        
        fig = go.Figure(data=[
            go.Bar(
                x=cluster_names,
                y=cluster_sizes,
                marker_color=colors,
                text=cluster_sizes,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             'Anzahl: %{y}<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Cluster",
            yaxis_title="Anzahl Punkte",
            showlegend=False,
            width=600,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_metrics_chart(
        self,
        metrics: Dict[str, Any],
        title: str = "Clustering-Metriken"
    ) -> go.Figure:
        """Erstellt ein Balkendiagramm der Metriken"""
        
        # Verfügbare Metriken sammeln
        metric_data = []
        metric_names = []
        metric_colors = []
        
        # Interne Metriken
        internal_metrics = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
        internal_colors = ['#2E8B57', '#DC143C', '#4169E1']  # Grün, Rot, Blau
        
        for i, metric in enumerate(internal_metrics):
            if metric in metrics and metrics[metric] is not None:
                metric_data.append(metrics[metric])
                metric_names.append(metric.replace('_', ' ').title())
                metric_colors.append(internal_colors[i])
        
        if not metric_data:
            # Fallback: Leere Figur
            fig = go.Figure()
            fig.update_layout(
                title=dict(text=title, x=0.5),
                xaxis_title="Metriken",
                yaxis_title="Wert",
                width=600,
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return fig
        
        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_data,
                marker_color=metric_colors,
                text=[f"{val:.3f}" for val in metric_data],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             'Wert: %{y:.4f}<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Metriken",
            yaxis_title="Wert",
            showlegend=False,
            width=600,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_silhouette_plot(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        title: str = "Silhouette-Analyse"
    ) -> go.Figure:
        """Erstellt einen Silhouette-Plot"""
        from sklearn.metrics import silhouette_samples
        
        try:
            silhouette_vals = silhouette_samples(X, labels)
            unique_labels = np.unique(labels)
            
            # Silhouette-Werte pro Cluster sammeln
            y_lower = 10
            colors = self._generate_colors(len(unique_labels))
            
            fig = go.Figure()
            
            for i, label in enumerate(unique_labels):
                if label == -1:  # Outlier überspringen
                    continue
                    
                cluster_silhouette_vals = silhouette_vals[labels == label]
                cluster_silhouette_vals.sort()
                
                size_cluster = cluster_silhouette_vals.shape[0]
                y_upper = y_lower + size_cluster
                
                fig.add_trace(go.Scatter(
                    x=cluster_silhouette_vals,
                    y=list(range(y_lower, y_upper)),
                    mode='markers',
                    name=f'Cluster {label}',
                    marker=dict(
                        color=colors[i],
                        size=4
                    ),
                    showlegend=True
                ))
                
                y_lower = y_upper + 10
            
            # Durchschnittliche Silhouette hinzufügen
            avg_silhouette = np.mean(silhouette_vals)
            fig.add_vline(
                x=avg_silhouette,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Durchschnitt: {avg_silhouette:.3f}"
            )
            
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=16)
                ),
                xaxis_title="Silhouette-Wert",
                yaxis_title="Cluster-Label",
                width=800,
                height=600,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            # Fallback: Fehler-Plot
            fig = go.Figure()
            fig.add_annotation(
                text=f"Silhouette-Plot konnte nicht erstellt werden: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title=dict(text=title, x=0.5),
                width=800, height=600,
                plot_bgcolor='white', paper_bgcolor='white'
            )
            return fig
    
    def create_elbow_plot(
        self,
        k_values: List[int],
        inertias: List[float],
        title: str = "Elbow-Methode"
    ) -> go.Figure:
        """Erstellt einen Elbow-Plot für K-Means"""
        
        fig = go.Figure(data=[
            go.Scatter(
                x=k_values,
                y=inertias,
                mode='lines+markers',
                name='Inertia',
                line=dict(color='blue', width=2),
                marker=dict(size=8, color='blue')
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Anzahl Cluster (k)",
            yaxis_title="Inertia",
            showlegend=False,
            width=600,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_combined_dashboard(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        metrics: Dict[str, Any],
        hover_data: Optional[pd.DataFrame] = None,
        hover_columns: Optional[List[str]] = None
    ) -> go.Figure:
        """Erstellt ein kombiniertes Dashboard mit mehreren Plots"""
        
        # Subplots erstellen
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Clustering Visualisierung",
                "Cluster-Größen",
                "Metriken",
                "Silhouette-Analyse"
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Scatter Plot
        unique_labels = np.unique(labels)
        colors = self._generate_colors(len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            hover_text = self._create_hover_text(mask, hover_data, hover_columns, label)
            
            if label == -1:
                label_name = "Outlier"
                marker_symbol = 'x'
            else:
                label_name = f"Cluster {label}"
                marker_symbol = 'circle'
            
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode='markers',
                    name=label_name,
                    marker=dict(
                        size=6,
                        color=colors[i],
                        opacity=0.7,
                        symbol=marker_symbol
                    ),
                    text=hover_text,
                    hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Cluster-Größen
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        cluster_names = [f"Cluster {label}" if label != -1 else "Outlier" for label in unique_labels]
        
        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=cluster_sizes,
                marker_color=colors,
                text=cluster_sizes,
                textposition='auto',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Metriken
        metric_data = []
        metric_names = []
        internal_metrics = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
        
        for metric in internal_metrics:
            if metric in metrics and metrics[metric] is not None:
                metric_data.append(metrics[metric])
                metric_names.append(metric.replace('_', ' ').title())
        
        if metric_data:
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_data,
                    marker_color=['#2E8B57', '#DC143C', '#4169E1'][:len(metric_data)],
                    text=[f"{val:.3f}" for val in metric_data],
                    textposition='auto',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Silhouette-Analyse (vereinfacht)
        try:
            from sklearn.metrics import silhouette_samples
            silhouette_vals = silhouette_samples(X, labels)
            avg_silhouette = np.mean(silhouette_vals)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(silhouette_vals))),
                    y=silhouette_vals,
                    mode='markers',
                    name='Silhouette',
                    marker=dict(size=2, color='blue', opacity=0.6),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Durchschnittslinie
            fig.add_hline(
                y=avg_silhouette,
                line_dash="dash",
                line_color="red",
                row=2, col=2
            )
        except:
            pass
        
        # Layout anpassen
        fig.update_layout(
            title=dict(
                text="Clustering Dashboard",
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Achsen-Labels
        fig.update_xaxes(title_text="Dimension 1", row=1, col=1)
        fig.update_yaxes(title_text="Dimension 2", row=1, col=1)
        fig.update_xaxes(title_text="Cluster", row=1, col=2)
        fig.update_yaxes(title_text="Anzahl", row=1, col=2)
        fig.update_xaxes(title_text="Metriken", row=2, col=1)
        fig.update_yaxes(title_text="Wert", row=2, col=1)
        fig.update_xaxes(title_text="Punkte", row=2, col=2)
        fig.update_yaxes(title_text="Silhouette", row=2, col=2)
        
        return fig
    
    def _generate_colors(self, n_colors: int) -> List[str]:
        """Generiert n verschiedene Farben"""
        if n_colors <= len(self.default_colors):
            return self.default_colors[:n_colors]
        
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.8)
            hex_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
            colors.append(hex_color)
        
        return colors
    
    def _create_hover_text(
        self,
        mask: np.ndarray,
        hover_data: Optional[pd.DataFrame],
        hover_columns: Optional[List[str]],
        label: int
    ) -> List[str]:
        """Erstellt Hover-Text für die Punkte"""
        if hover_data is None or hover_columns is None:
            return [f"Index: {i}" for i in np.where(mask)[0]]
        
        hover_texts = []
        for i, idx in enumerate(np.where(mask)[0]):
            text_parts = [f"Index: {idx}"]
            for col in hover_columns[:5]:  # Maximal 5 Spalten
                if col in hover_data.columns:
                    value = hover_data.iloc[idx][col]
                    if isinstance(value, float):
                        text_parts.append(f"{col}: {value:.3f}")
                    else:
                        text_parts.append(f"{col}: {value}")
            hover_texts.append("<br>".join(text_parts))
        
        return hover_texts
