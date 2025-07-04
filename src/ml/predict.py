import joblib
import pandas as pd
import numpy as np
import os
import sys
from src.preprocesamiento.preprocessing import preprocess_input_data

# --- Configuración de rutas para importar módulos ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

# --- Rutas a los archivos de modelos y transformadores guardados ---
MODELS_DIR = os.path.join(project_root, 'src', 'models')
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, 'final_model.joblib')
ONEHOT_ENCODER_PATH = os.path.join(MODELS_DIR, 'onehot_encoder.joblib')
MINMAX_SCALER_PATH = os.path.join(MODELS_DIR, 'minmax_scaler.joblib')
LABEL_ENCODER_TARGET_PATH = os.path.join(MODELS_DIR, 'label_encoder.joblib')

# --- Variables globales para almacenar los activos cargados ---
_model = None
_onehot_encoder = None
_minmax_scaler = None
_label_encoder_target = None

# --- Cargar archivos de transformacion y modelo entrenado ---
def _load_assets():
    """
    Carga el modelo y los transformadores si aún no están cargados.
    """
    global _model, _onehot_encoder, _minmax_scaler, _label_encoder_target
    if _model is None or _onehot_encoder is None or _minmax_scaler is None or _label_encoder_target is None:
        try:
            # Verificar si los archivos existen
            if not os.path.exists(FINAL_MODEL_PATH):
                raise FileNotFoundError(f"Archivo del modelo no encontrado en {FINAL_MODEL_PATH}.")
            if not os.path.exists(ONEHOT_ENCODER_PATH):
                raise FileNotFoundError(f"Archivo del OneHotEncoder no encontrado en {ONEHOT_ENCODER_PATH}.")
            if not os.path.exists(MINMAX_SCALER_PATH):
                raise FileNotFoundError(f"Archivo del MinMaxScaler no encontrado en {MINMAX_SCALER_PATH}.")
            if not os.path.exists(LABEL_ENCODER_TARGET_PATH):
                raise FileNotFoundError(f"Archivo del LabelEncoder (target) no encontrado en {LABEL_ENCODER_TARGET_PATH}.")

            # Cargar los componentes
            _model = joblib.load(FINAL_MODEL_PATH)
            _onehot_encoder = joblib.load(ONEHOT_ENCODER_PATH)
            _minmax_scaler = joblib.load(MINMAX_SCALER_PATH)
            _label_encoder_target = joblib.load(LABEL_ENCODER_TARGET_PATH)
            print("Modelo y transformadores cargados exitosamente!")
        except Exception as e:
            raise RuntimeError(f"Error al cargar los activos del modelo: {e}")
    return _model, _onehot_encoder, _minmax_scaler, _label_encoder_target

# --- Realizar la prediccion segun la informacion recibida ---
def make_prediction(input_data: dict):
    
    model, ohe, mms, label_encoder_target = _load_assets()

    try:
        # Convertir el diccionario de entrada a un DataFrame de Pandas.
        input_df = pd.DataFrame([input_data])

        # Transformar los datos de entrada
        processed_input_df = preprocess_input_data(input_df, ohe, mms)

        # --- Validar que el processed_input_df tenga las columnas correctas para el modelo ---
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
             processed_input_df = processed_input_df[model.feature_names_in_]
        else:
            print("Advertencia: El modelo no expone feature_names_in_. Asegúrate de que las columnas de entrada al modelo estén en el orden y con los nombres correctos.")
    except KeyError as e:
        raise ValueError(f"Faltan características de entrada en el JSON: {e}. "
                         "Asegúrate de enviar todas las características esperadas por los transformadores "
                         "antes de la pre-transformación (ej. 'Género', 'Edad', 'Distrito', etc.).")
    except Exception as e:
        raise ValueError(f"Error durante el preprocesamiento de los datos de entrada: {e}. "
                         "Verifica el formato y los tipos de tus datos.")

    # Realizar la predicción con los datos preprocesados
    prediction_probabilities_all = model.predict_proba(processed_input_df)[0] 
    prediction_probabilities_all = np.array(prediction_probabilities_all)

    # Obtener los índices y probabiliades de las 3 clases con mayor probabilidad
    top_3_indices = prediction_probabilities_all.argsort()[-3:][::-1] 
    top_3_probabilities = prediction_probabilities_all[top_3_indices]

    # Normalizamos las probabilidades
    total = np.sum(top_3_probabilities)
    normalized_percentages = [round((p / total) * 100.0, 2) for p in top_3_probabilities]

    # Decodificar las etiquetas
    top_3_careers = []
    for idx, percentage in zip(top_3_indices, normalized_percentages):
        if label_encoder_target:
            career_name = str(label_encoder_target.inverse_transform([idx])[0])
        else:
            if hasattr(model, 'classes_') and idx < len(model.classes_):
                career_name = str(model.classes_[idx])
            else:
                career_name = f"Clase ID: {int(idx)}"
        
        top_3_careers.append({
            "nameCareer": career_name,
            "hitRate": percentage
        })

    return top_3_careers
