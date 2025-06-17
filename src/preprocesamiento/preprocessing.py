# src/preprocesamiento/preprocessing.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import os

clasificacion_socioeconomica_inference = { # Para crear 'nivel_socioeconomico' a partir de 'Distrito'
    "A": ["San Isidro", "La Molina", "Miraflores", "San Borja"],
    "B": ["Jesús María", "Magdalena del Mar", "Santiago de Surco", "Lince", "Barranco"],
    "C": ["Pueblo Libre", "San Miguel", "Chorrillos", "Cercado de Lima", "Rímac", "Santa Anita", "Breña", "Surquillo", "San Luis"],
    "D": ["Comas", "Los Olivos", "Independencia", "Villa el Salvador", "San Juan de Miraflores", "El Agustino", "La Victoria", "Lurín", "Callao", "callao", "Lurigancho"],
    "E": ["Villa Maria del Triunfo", "San Juan de Lurigancho", "Ate", "Puente Piedra", "Carabayllo", "San Martin de Porres", "Pachacámac"]
}

def _assign_socioeconomic_level(distrito):
    for nivel, distritos in clasificacion_socioeconomica_inference.items():
        if distrito in distritos:
            return nivel
    return "No clasificado"


def preprocess_input_data(df: pd.DataFrame, ohe: OneHotEncoder, mms: MinMaxScaler):
    
    # Asignar nivel socioeconómico si 'Distrito' está presente
    if 'Distrito' in df.columns:
        df["nivel_socioeconomico"] = df["district"].apply(_assign_socioeconomic_level)
    else:
        df["nivel_socioeconomico"] = "No clasificado"

    # Definir columnas para las transformaciones 
    categorical_cols = ['genre', 'preferred_course_1', 'preferred_course_2', 'preferred_course_3',
                        'type_school', 'area', 'nivel_socioeconomico']
    numeric_cols = ['empathy_level', 'listen_level', 'solution_level',
                    'communication_level', 'teamwork_level', 'monthly_cost']
    
    # Transformación de columnas categóricas
    encoded_data = ohe.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(categorical_cols), index=df.index)

    # Transformación de columnas numéricas
    scaled_data = mms.transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=df.index)

    # Unión de los datos transformados
    transformed_df = pd.concat([encoded_df, scaled_df], axis=1)

    return transformed_df