import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import sys
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    r2_score
)

# Añadir el directorio raíz para permitir importaciones desde preprocesamiento y ml
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir)) # main.py ya está en src/
sys.path.append(project_root)

def extract():

    dataset_path = os.path.join(current_dir, 'data', 'dataset.xlsx') # current_dir es src/
    try:
        df = pd.read_excel(dataset_path)
        print(f"Dataset cargado exitosamente desde {dataset_path}")
    except FileNotFoundError:
        print(f"Error: dataset.xlsx no encontrado en {dataset_path}")

    return df


def cast_clean(df):

    #Se define los nuevos nombres de las columnas
    rename_columns = {
        'Hora de inicio': 'Hora_Inicio',
        'Hora de finalización': 'Hora_Finalizacion',
        'Correo electrónico': 'Correo_Electrónico',
        'Hora de la última modificación': 'Hora_Ultima_Modificacion',
        'Género': 'genre',
        'Distrito': 'district',
        '¿Cuál era tu curso preferido en la escuela? Primer Curso': 'preferred_course_1',
        '¿Cuál era tu curso preferido en la escuela? Segundo Curso': 'preferred_course_2',
        '¿Cuál era tu curso preferido en la escuela? Tercer Curso': 'preferred_course_3',
        'Tipo de Institución Escolar': 'type_school',
        '¿Qué tan empático te consideras?': 'empathy_level',
        '¿Qué tan bueno te consideras escuchando a otras personas?': 'listen_level',
        '¿Qué tan bueno te consideras solucionando problemas complejos?': 'solution_level',
        '¿Qué tan asertivo  te consideras al comunicarte con otras personas?': 'communication_level',
        '¿Qué tan bueno te consideras al trabajar en equipo?': 'teamwork_level',
        '¿Cuánto paga mensualmente por sus estudios universitarios?': 'monthly_cost',
        '¿Qué porcentaje de beca tiene asignada?': 'Porcentaje_Beca',
        '¿A que facultad perteneces?': 'Facultad',
        '¿Qué carrera estudias actualmente?': 'Carrera',
        '¿En que ciclo te encuentras?': 'Ciclo',
        '¿En que universidad estudias actualmente?': 'Universidad',
        '¿Se compromete a indicar que la información brindada en este formulario es verídica?': 'Compromiso',
        '¿Qué factores influyeron para que escojas la que estudias actualmente?': 'Factores'
    }
    
    #Aplicacion del casteo
    df.rename(columns=rename_columns, inplace=True)

    #Relaccion area - carrera
    area_mapping = {
        "Ingeniería de Sistemas de Información": "I", 
        "Ingeniería Civil": "I",
        "Ingeniería Industrial": "I",
        "Ingeniería Mecatrónica": "I",
        "Ingeniería Electrónica": "I",
        "Ingeniería Ambiental": "E",
        "Arquitectura": "A",
        "Economía": "C",
        "Marketing": "C",
        "Administración": "C",
        "Ingeniería de Gestión Empresarial": "C",
        "Contabilidad": "C",
        "Derecho": "H",
        "Ciencias de la Comunicación": "H",
        "Educación": "H",
        "Psicología": "H",
        "Diseño Gráfico": "A",
        "Odontología": "S",
        "Medicina": "S",
        "Ingeniería de Minas": "I",
        "Gastronomía": "A",
        "Ingeniería Agroindustrial": "E",
        "Arte": "A",
        "Medicina Veterinaria": "S",
        "Ingeniería Biomédica": "I",
        "Turismo": "H",
        "Ingeniería de Telecomunicaciones": "I",
        "Diseño de Modas": "A",
        "Enfermería": "S",
    }

    df["area"] = df["Carrera"].map(area_mapping)

    #Relaccion area2 - carrera
    area_mapping2 = {
        "Ingeniería de Sistemas de Información": "C", 
        "Ingeniería Civil": "E",
        "Ingeniería Industrial": "C",
        "Ingeniería Mecatrónica": "E",
        "Ingeniería Electrónica": "E",
        "Ingeniería Ambiental": "I",
        "Arquitectura": "I",
        "Economía": "H",
        "Marketing": "H",
        "Administración": "H",
        "Ingeniería de Gestión Empresarial": "I",
        "Contabilidad": "I",
        "Derecho": "C",
        "Ciencias de la Comunicación": "A",
        "Educación": "C",
        "Psicología": "S",
        "Diseño Gráfico": "H",
        "Odontología": "H",
        "Medicina": "H",
        "Ingeniería de Minas": "E",
        "Gastronomía": "S",
        "Ingeniería Agroindustrial": "C",
        "Arte": "H",
        "Medicina Veterinaria": "E",
        "Ingeniería Biomédica": "S",
        "Turismo": "C",
        "Ingeniería de Telecomunicaciones": "E",
        "Diseño de Modas": "C",
        "Enfermería": "H",
    }

    df["area2"] = df["Carrera"].map(area_mapping2)
    
    #Definicion de escalas socioeconomicas segun distrito
    clasificacion_socioeconomica = {
        "A": ["San Isidro", "La Molina", "Miraflores", "San Borja"],
        "B": ["Jesús María", "Magdalena del Mar", "Santiago de Surco", "Lince", "Barranco"],
        "C": ["Pueblo Libre", "San Miguel", "Chorrillos", "Cercado de Lima", "Rímac",
               "Santa Anita", "Breña", "Surquillo", "San Luis"],
        "D": ["Comas", "Los Olivos", "Independencia", "Villa el Salvador", "San Juan de Miraflores", 
              "El Agustino", "La Victoria", "Lurín", "Callao", "callao", "Lurigancho"],
        "E": ["Villa Maria del Triunfo", "San Juan de Lurigancho", "Ate", "Puente Piedra", 
              "Carabayllo", "San Martin de Porres", "Pachacámac"]
    }

    # Función para asignar nivel socioeconómico
    def asignar_nivel(distrito):
        for nivel, distritos in clasificacion_socioeconomica.items():
            if distrito in distritos:
                return nivel
        return "No clasificado"

    # Aplicar la función al DataFrame
    df["nivel_socioeconomico"] = df["district"].apply(asignar_nivel)

    #Eliminamos columnas innecesarias o sin impacto en el target
    df.drop(columns='Hora_Ultima_Modificacion',inplace=True)
    df.drop(columns='Nombre',inplace=True)
    df.drop(columns='Correo_Electrónico',inplace=True)
    df.drop(columns='ID',inplace=True)
    df.drop(columns='Hora_Finalizacion',inplace=True)
    df.drop(columns='Hora_Inicio',inplace=True)
    df.drop(columns='Compromiso',inplace=True)
    df.drop(columns='Nombre Completo',inplace=True)
    df.drop(columns='Correo electrónico de contacto',inplace=True)
    df.drop(columns='Universidad',inplace=True)
    df.drop(columns='Ciclo',inplace=True)
    df.drop(columns='Porcentaje_Beca',inplace=True)

    return df


def pipeline_train_model():
    """
    Ejecuta el pipeline completo de Machine Learning:
    1. Carga y limpia el dataset.
    2. Entrena y guarda los transformadores.
    3. Entrena y guarda el modelo.
    """

    models_dir = os.path.join(current_dir, 'models') 
    os.makedirs(models_dir, exist_ok=True)

    # 1. Cargar y limpiar los datos (Usamos las funciones definidas aquí)
    df = extract()
    df_clean = cast_clean(df)

    # 2. Definir columnas para la transformación
    categorical_cols = ['genre', 'preferred_course_1', 'preferred_course_2', 'preferred_course_3',
                        'type_school', 'area', 'area2', 'nivel_socioeconomico']
    numeric_cols = ['empathy_level', 'listen_level', 'solution_level',
                    'communication_level', 'teamwork_level', 'monthly_cost']
    target_col = 'Carrera' # El nombre original del target

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # --- 3. Entrenar y guardar los transformadores ---
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    ohe.fit(X[categorical_cols])
    joblib.dump(ohe, os.path.join(models_dir, 'onehot_encoder.joblib'))
    print(f"OneHotEncoder guardado en {os.path.join(models_dir, 'onehot_encoder.joblib')}")

    mms = MinMaxScaler()
    mms.fit(X[numeric_cols])
    joblib.dump(mms, os.path.join(models_dir, 'minmax_scaler.joblib'))
    print(f"MinMaxScaler guardado en {os.path.join(models_dir, 'minmax_scaler.joblib')}")

    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.joblib'))
    print(f"LabelEncoder (target) guardado en {os.path.join(models_dir, 'label_encoder.joblib')}")
    
    y_encoded = label_encoder.transform(y) # Codificar el target para el entrenamiento

    # --- 4. Aplicar las transformaciones a los datos de entrenamiento ---
    # Usamos las transformaciones aprendidas para transformar X
    encoded_categorical_data = ohe.transform(X[categorical_cols])
    encoded_categorical_df = pd.DataFrame(encoded_categorical_data,
                                          columns=ohe.get_feature_names_out(categorical_cols),
                                          index=X.index)

    scaled_numeric_data = mms.transform(X[numeric_cols])
    scaled_numeric_df = pd.DataFrame(scaled_numeric_data, columns=numeric_cols, index=X.index)

    # Concatenar para obtener el X_processed final para el modelo
    X_processed = pd.concat([encoded_categorical_df, scaled_numeric_df], axis=1)

    # --- 5. Preparar datos para el entrenamiento del modelo (train_test_split, oversampling) ---
    X_train, X_test, Y_train, Y_test = train_test_split(X_processed, y_encoded, test_size=0.3, random_state=42)

    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, Y_train)

    # --- 6. Entrenar y guardar el Modelo ---
    model_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=1
    )
    model_rf.fit(X_resampled, y_resampled)
    joblib.dump(model_rf, os.path.join(models_dir, 'final_model.joblib'))
    print(f"Modelo final (RandomForestClassifier) guardado en {os.path.join(models_dir, 'final_model.joblib')}")

    #Probamos con los datos de prueba
    y_pred_RandFor = model_rf.predict(X_test)
    y_proba = model_rf.predict_proba(X_test)

    # Métricas básicas
    print("📊 Métricas de Evaluación:")
    print("Accuracy:", accuracy_score(Y_test, y_pred_RandFor))
    print("Precision (macro):", precision_score(Y_test, y_pred_RandFor, average='macro'))
    print("Recall (macro):", recall_score(Y_test, y_pred_RandFor, average='macro'))
    print("F1-score (macro):", f1_score(Y_test, y_pred_RandFor, average='macro'))
    print("Cohen's Kappa:", cohen_kappa_score(Y_test, y_pred_RandFor))
    roc_auc = roc_auc_score(Y_test, y_proba, multi_class='ovr', average='macro')
    print("ROC AUC (multiclase, OVR):", roc_auc)
    r2 = r2_score(Y_test, y_pred_RandFor)
    print("R² en regresión lineal:", r2)


if __name__ == "__main__":
    pipeline_train_model()