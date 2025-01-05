import pandas as pd
import numpy as np

def process_data(df: pd.DataFrame, columns_to_impute: list, target_column: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Procesa los datos:
    - Imputa los valores faltantes
    - Escalar las variables numéricas

    Args:
       df (pd.DataFrame): DataFrame con los datos a procesar.
       columns_to_impute (list): Lista de columnas a imputar los valores faltantes.
       target_column (str, optional): Nombre de la columna objetivo.

    Returns:
       pd.DataFrame: DataFrame con los datos procesados.
       pd.Series: Serie con la variable objetivo.
    """
    # Reemplazar valores 0 por NaN en las columnas a imputar
    df[columns_to_impute] = df[columns_to_impute].replace(0, np.nan)
    
    # Extraer la columna objetivo si es proporcionada
    target = df[target_column] if target_column else None
    
    return df, target
