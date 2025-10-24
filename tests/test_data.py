"""
Tests für das Datenmodul
"""

import pytest
import pandas as pd
import numpy as np
from app.data import DataProcessor, create_sample_data


class TestDataProcessor:
    """Tests für DataProcessor"""
    
    def test_init(self):
        """Test Initialisierung"""
        processor = DataProcessor(random_state=42)
        assert processor.random_state == 42
        assert processor.scaler is None
        assert processor.feature_selector is None
    
    def test_create_sample_data(self):
        """Test Beispieldaten-Erstellung"""
        df = create_sample_data(n_samples=100, n_features=5, n_clusters=3)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 100
        assert df.shape[1] >= 5  # Mindestens 5 Features
        assert 'true_cluster' in df.columns
    
    def test_analyze_columns(self):
        """Test Spalten-Analyse"""
        # Test-Daten erstellen
        df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'boolean': [True, False, True, False, True],
            'text': ['text1', 'text2', 'text3', 'text4', 'text5']
        })
        
        processor = DataProcessor()
        analysis = processor.analyze_columns(df)
        
        assert 'numeric_columns' in analysis
        assert 'categorical_columns' in analysis
        assert 'boolean_columns' in analysis
        assert 'text_columns' in analysis
        assert 'numeric1' in analysis['numeric_columns']
        assert 'numeric2' in analysis['numeric_columns']
        assert 'categorical' in analysis['categorical_columns']
        assert 'boolean' in analysis['boolean_columns']
    
    def test_preprocess_data(self):
        """Test Datenverarbeitung"""
        # Test-Daten mit Missing Values
        df = pd.DataFrame({
            'numeric1': [1, 2, np.nan, 4, 5],
            'numeric2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
        
        processor = DataProcessor(random_state=42)
        
        # Preprocessing durchführen
        df_processed = processor.preprocess_data(
            df,
            selected_columns=['numeric1', 'numeric2', 'categorical'],
            handle_missing='impute',
            imputation_method='mean',
            scaling_method='standard',
            one_hot_encode=True,
            remove_duplicates=False
        )
        
        assert isinstance(df_processed, pd.DataFrame)
        assert df_processed.isnull().sum().sum() == 0  # Keine Missing Values
        assert len(df_processed.columns) > 3  # One-Hot Encoding sollte Spalten hinzufügen
    
    def test_sample_data(self):
        """Test Stichproben-Erstellung"""
        df = pd.DataFrame({
            'col1': range(100),
            'col2': range(100, 200)
        })
        
        processor = DataProcessor(random_state=42)
        
        # 50% Stichprobe
        df_sampled = processor.sample_data(df, sample_size=0.5)
        
        assert len(df_sampled) == 50
        assert df_sampled.shape[1] == 2
    
    def test_get_data_summary(self):
        """Test Daten-Zusammenfassung"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        processor = DataProcessor()
        summary = processor.get_data_summary(df)
        
        assert 'shape' in summary
        assert 'memory_usage' in summary
        assert 'missing_values' in summary
        assert summary['shape'] == (5, 2)
        assert summary['missing_values'] == 0
