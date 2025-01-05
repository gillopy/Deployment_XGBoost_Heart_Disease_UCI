import pandas as pd
import numpy as np

def process_data(df: pd.DataFrame, columns_to_impute: list, target_column: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Procesa los datos:
    - Imputa los valores faltantes
    - Elimina las filas con valores NaN y asegura la correspondencia con la columna objetivo

    Args:
       df (pd.DataFrame): DataFrame con los datos a procesar.
       columns_to_impute (list): Lista de columnas en las que se imputarán valores faltantes.
       target_column (str): Nombre de la columna objetivo (opcional).

    Returns:
       tuple: (DataFrame procesado, Series con la columna objetivo si se proporciona).
    """
    # Reemplazar ceros por NaN en las columnas especificadas
    df[columns_to_impute] = df[columns_to_impute].replace(0, np.nan)
    
    # Separar la columna objetivo si se especifica
    target = df[target_column] if target_column else None
    
    # Imprimir forma original del DataFrame
    print(f"Original shape of DataFrame: {df.shape}")
    
    # Eliminar filas con valores NaN en el DataFrame
    df_cleaned = df.dropna()
    print(f"Shape after dropping NaN: {df_cleaned.shape}")
    
    # Si hay una columna objetivo, ajustarla para que coincida con los índices de df_cleaned
    if target_column:
        target = target.loc[df_cleaned.index]
        print(f"Shape of target column after cleaning: {target.shape}")
    
    return df_cleaned, target

