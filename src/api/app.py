# src/api/app.py

from flask import Flask, request, jsonify
import sys
import os

# --- Configuración de rutas para importar módulos ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) # La raíz del proyecto
sys.path.append(project_root)

# Importa la función de predicción y la función de carga de activos
from src.ml.predict import make_prediction, _load_assets

app = Flask(__name__)

# --- Intentar cargar el modelo y transformadores al inicio de la aplicación ---
try:
    _load_assets()
    print("Aplicación Flask iniciada y componentes del modelo cargados.")
except (FileNotFoundError, RuntimeError) as e:
    print(f"ERROR CRÍTICO DURANTE LA INICIALIZACIÓN DE LA APLICACIÓN: {e}")
    print("La aplicación continuará, pero las predicciones fallarán hasta que se resuelva el problema del modelo/transformadores.")

# --- Endpoint de prueba ---
@app.route('/', methods=['GET'])
def home():
    return "¡Bienvenido a la API de Predicción de Carreras!"

# --- Endpoint de Predicción ---
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type debe ser application/json"}), 400

        data = request.get_json()
        
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Payload JSON inválido. Se espera un diccionario con las características de entrada."}), 400

        prediction_result = make_prediction(data)

        return jsonify(prediction_result)

    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as e:
        print(f"Error de datos de entrada/carga de activos en /predict: {e}")
        return jsonify({"error": str(e), "message": "Fallo en la predicción. Verifica los datos de entrada o la disponibilidad del modelo."}), 400
    except Exception as e:
        print(f"Error inesperado en /predict: {e}")
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)