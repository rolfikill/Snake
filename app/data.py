"""
Datenmodul für Upload, Typing und Preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import chardet
import io


class DataProcessor:
    """Klasse für Datenverarbeitung und Preprocessing"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = None
        self.feature_selector = None
        self.column_types = {}
        self.original_columns = []
        
    def detect_encoding(self, file_path: str) -> str:
        """Erkennt die Encoding einer Datei"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Erste 10KB lesen
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'
    
    def detect_separator(self, file_path: str, encoding: str = 'utf-8') -> str:
        """Erkennt den Separator einer CSV-Datei"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                
            # Teste verschiedene Separatoren
            separators = [',', ';', '\t', '|']
            max_cols = 0
            best_sep = ','
            
            for sep in separators:
                cols = len(first_line.split(sep))
                if cols > max_cols:
                    max_cols = cols
                    best_sep = sep
                    
            return best_sep
        except Exception:
            return ','
    
    def load_data(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """Lädt Daten aus CSV oder Parquet Datei"""
        try:
            if file_type.lower() == 'csv':
                encoding = self.detect_encoding(file_path)
                separator = self.detect_separator(file_path, encoding)
                
                df = pd.read_csv(
                    file_path, 
                    encoding=encoding, 
                    sep=separator,
                    low_memory=False
                )
            elif file_type.lower() == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            self.original_columns = df.columns.tolist()
            return df
            
        except Exception as e:
            raise Exception(f"Fehler beim Laden der Datei: {str(e)}")
    
    def load_data_from_bytes(self, file_bytes: bytes, file_name: str) -> pd.DataFrame:
        """Lädt Daten aus Bytes (für Streamlit Upload)"""
        try:
            file_extension = file_name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Versuche verschiedene Encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(
                            io.BytesIO(file_bytes),
                            encoding=encoding,
                            low_memory=False
                        )
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    raise ValueError("Konnte Datei nicht mit verfügbaren Encodings lesen")
                    
            elif file_extension == 'parquet':
                df = pd.read_parquet(io.BytesIO(file_bytes))
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
            self.original_columns = df.columns.tolist()
            return df
            
        except Exception as e:
            raise Exception(f"Fehler beim Laden der Datei: {str(e)}")
    
    def analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert Spalten und deren Typen"""
        analysis = {
            'numeric_columns': [],
            'categorical_columns': [],
            'boolean_columns': [],
            'datetime_columns': [],
            'text_columns': [],
            'missing_values': {},
            'column_info': {}
        }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'memory_usage': df[col].memory_usage(deep=True)
            }
            
            analysis['column_info'][col] = col_info
            analysis['missing_values'][col] = col_info['null_count']
            
            # Typ-Klassifizierung
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].dtype == 'bool' or df[col].nunique() == 2:
                    analysis['boolean_columns'].append(col)
                else:
                    analysis['numeric_columns'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                analysis['datetime_columns'].append(col)
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                if df[col].nunique() < 50:  # Kategorisch wenn < 50 eindeutige Werte
                    analysis['categorical_columns'].append(col)
                else:
                    analysis['text_columns'].append(col)
        
        self.column_types = analysis
        return analysis
    
    def preprocess_data(
        self, 
        df: pd.DataFrame,
        selected_columns: List[str],
        handle_missing: str = 'drop',
        imputation_method: str = 'mean',
        scaling_method: str = 'standard',
        one_hot_encode: bool = True,
        max_categories: int = 50,
        remove_duplicates: bool = True,
        variance_threshold: float = 0.0
    ) -> pd.DataFrame:
        """Führt Preprocessing der Daten durch"""
        
        # Kopie erstellen
        df_processed = df.copy()
        
        # Nur ausgewählte Spalten verwenden
        df_processed = df_processed[selected_columns]
        
        # Duplikate entfernen
        if remove_duplicates:
            initial_rows = len(df_processed)
            df_processed = df_processed.drop_duplicates()
            removed_rows = initial_rows - len(df_processed)
            if removed_rows > 0:
                print(f"Entfernt {removed_rows} Duplikate")
        
        # Missing Values behandeln
        if handle_missing == 'drop':
            df_processed = df_processed.dropna()
        elif handle_missing == 'impute':
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        if imputation_method == 'mean':
                            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                        elif imputation_method == 'median':
                            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                        elif imputation_method == 'most_frequent':
                            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                    else:
                        # Für kategorische Spalten
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        
        # Datetime zu numerisch konvertieren
        for col in df_processed.columns:
            if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                df_processed[col] = pd.to_numeric(df_processed[col])
        
        # One-Hot Encoding für kategorische Spalten
        if one_hot_encode:
            categorical_cols = []
            for col in df_processed.columns:
                if (df_processed[col].dtype == 'object' and 
                    df_processed[col].nunique() <= max_categories):
                    categorical_cols.append(col)
            
            if categorical_cols:
                df_processed = pd.get_dummies(
                    df_processed, 
                    columns=categorical_cols, 
                    prefix=categorical_cols,
                    drop_first=True
                )
        
        # Nur numerische Spalten behalten
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed = df_processed[numeric_cols]
        
        # Feature Selection basierend auf Varianz
        if variance_threshold > 0:
            self.feature_selector = VarianceThreshold(threshold=variance_threshold)
            df_processed = pd.DataFrame(
                self.feature_selector.fit_transform(df_processed),
                columns=df_processed.columns[self.feature_selector.get_support()],
                index=df_processed.index
            )
        
        # Skalierung
        if scaling_method != 'none':
            if scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaling_method == 'robust':
                self.scaler = RobustScaler()
            
            df_processed = pd.DataFrame(
                self.scaler.fit_transform(df_processed),
                columns=df_processed.columns,
                index=df_processed.index
            )
        
        return df_processed
    
    def sample_data(self, df: pd.DataFrame, sample_size: float = 1.0) -> pd.DataFrame:
        """Erstellt eine Stichprobe der Daten"""
        if sample_size >= 1.0:
            return df
        
        n_samples = int(len(df) * sample_size)
        return df.sample(n=n_samples, random_state=self.random_state)
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Erstellt eine Zusammenfassung der Daten"""
        return {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns)
        }


def create_sample_data(n_samples: int = 1000, n_features: int = 10, n_clusters: int = 3) -> pd.DataFrame:
    """Erstellt Beispieldaten für Tests"""
    from sklearn.datasets import make_blobs, make_moons, make_circles
    
    # Verschiedene Datensätze erstellen
    datasets = []
    
    # Blobs
    X_blobs, y_blobs = make_blobs(
        n_samples=n_samples//3, 
        centers=n_clusters, 
        n_features=n_features//2,
        random_state=42
    )
    df_blobs = pd.DataFrame(X_blobs, columns=[f'blob_feature_{i}' for i in range(X_blobs.shape[1])])
    df_blobs['true_cluster'] = y_blobs
    datasets.append(df_blobs)
    
    # Moons
    X_moons, y_moons = make_moons(
        n_samples=n_samples//3, 
        noise=0.1, 
        random_state=42
    )
    df_moons = pd.DataFrame(X_moons, columns=[f'moon_feature_{i}' for i in range(X_moons.shape[1])])
    df_moons['true_cluster'] = y_moons
    datasets.append(df_moons)
    
    # Zusätzliche numerische Features
    np.random.seed(42)
    n_extra_features = n_features - X_blobs.shape[1] - X_moons.shape[1]
    if n_extra_features > 0:
        extra_data = np.random.randn(n_samples//3, n_extra_features)
        df_extra = pd.DataFrame(
            extra_data, 
            columns=[f'extra_feature_{i}' for i in range(n_extra_features)]
        )
        datasets.append(df_extra)
    
    # Alle Datensätze zusammenführen
    df_combined = pd.concat(datasets, axis=1)
    
    # Einige kategorische Features hinzufügen
    df_combined['category_A'] = np.random.choice(['A', 'B', 'C'], n_samples//3)
    df_combined['category_B'] = np.random.choice(['X', 'Y'], n_samples//3)
    
    # Einige Missing Values hinzufügen
    missing_indices = np.random.choice(df_combined.index, size=int(0.05 * len(df_combined)), replace=False)
    df_combined.loc[missing_indices, 'blob_feature_0'] = np.nan
    
    return df_combined
