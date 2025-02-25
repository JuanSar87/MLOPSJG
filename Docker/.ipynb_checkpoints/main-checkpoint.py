from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="API de Inferencia de Modelos")

# Cargar los modelos
model_paths = {
    "random_forest": "models/random_forest_model.pkl",
    "logistic_regression": "models/logistic_regression_model.pkl",
    "logistic_regression1": "models/logistic_regression_model.pkl"
}

models = {name: joblib.load(path) for name, path in model_paths.items()}

# Variable global para almacenar el modelo seleccionado
selected_model = "random_forest"

label_mapping = {0: "Female", 1: "Male"}

# Definir esquema de entrada para la API
class PredictionInput(BaseModel):
    features: list[float]

class ModelSelectionInput(BaseModel):
    model_name: str

# Ruta de prueba
@app.get("/")
def home():
    return {"message": "API de Inferencia Activa"}

# Ruta para la interfaz web
@app.get("/interface", response_class=HTMLResponse)
def get_interface():
    return """
    <html>
    <head>
        <title>Inferencia de Modelos</title>
    </head>
    <body>
        <h2>Selecciona el Modelo:</h2>
        <select id="modelSelect">
            <option value="random_forest">Random Forest</option>
            <option value="logistic_regression">Logistic Regression</option>
            <option value="logistic_regression1">Logistic Regression_bien</option>
        </select>

        <h2>Ingresa las Características (separadas por comas):</h2>
        <input type="text" id="featuresInput" placeholder="Ej: 5.1, 3.5, 1.4, 0.2" />
        <br><br>
        <button onclick="makePrediction()">Predecir</button>

        <h2>Resultado:</h2>
        <input type="text" id="result" readonly style="font-weight: bold; color: blue;" />

        <script>
            async function makePrediction() {
                const model = document.getElementById("modelSelect").value;
                const features = document.getElementById("featuresInput").value.split(",").map(Number);

                // Seleccionar el modelo
                await fetch("/set_model/", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ model_name: model })
                });

                // Hacer la predicción
                const response = await fetch("/predict/", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ features: features })
                });

                const data = await response.json();
                document.getElementById("result").value = data.predicted_label;
            }
        </script>
    </body>
    </html>
    """

# Endpoint para seleccionar el modelo manualmente
@app.post("/set_model/")
def set_model(input_data: ModelSelectionInput):
    global selected_model

    model_name = input_data.model_name.lower()
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Modelo no válido. Usa 'random_forest' o 'logistic_regression'.")

    selected_model = model_name
    return {"message": f"Modelo seleccionado: {selected_model}"}

# Endpoint para hacer predicciones usando el modelo seleccionado
@app.post("/predict/")
def predict(input_data: PredictionInput):
    model = models[selected_model]

    # Convertir los datos de entrada en un array de NumPy
    features_array = np.array([input_data.features])

    # Predicción
    prediction = model.predict(features_array)[0]
    prediction_label = label_mapping.get(int(prediction), "Unknown")

    return {
        "model_used": selected_model,
        "input_features": input_data.features,
        "predicted_label": prediction_label
    }
