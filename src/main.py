import sys
import os
# Agregar la raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_loader import load_data
from src.data.data_processor import process_data
import pandas as pd
from src.data.data_splitter import split_data
from src.model.trainer import train_model
from src.model.evaluator import evaluate_model
from src.model.saver import save_model

def main():
    
    # Cargar los datos
    heart_disease = load_data(file_path = "data/raw/heart_disease.csv")
    
    # Preprocesar los datos
    processed_data, target = process_data(
                                  df=heart_disease, 
                                  columns_to_impute=['trestbps', 'chol', 'thalach', 'oldpeak'],  # columnas que pueden tener NaN
                                  target_column='num'  # Cambio para coincidir con la columna objetivo del dataset
                                  )
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(processed_data, target_column='num')
    
    # Convertir 'y' en binario
    y_train_binary = (y_train > 0).astype(int)
    y_test_binary = (y_test > 0).astype(int)

    # Entrenar el modelo
    model = train_model(X_train=X_train, y_train=y_train_binary)



    # Evaluar el modelo
    accuracy, precision, recall, f1, auc = evaluate_model(model, test_data=X_test, y_test=y_test_binary)

    # Imprimir las métricas
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"AUC: {auc:.3f}")

    
    # Guardar el modelo
    save_model(model, model_path="models/trained_model")
    
if __name__ == "__main__":
    main()
