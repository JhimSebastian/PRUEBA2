from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from ultralytics import YOLO
import base64

app = Flask(__name__)

# Directorio base
directorio_base = os.path.dirname(os.path.abspath(__file__))

# Ruta del modelo
ruta_modelo = os.path.join(directorio_base, "models_movil", "best110.pt")

if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"El modelo no se encontró en: {ruta_modelo}")

# Cargar el modelo YOLO en CPU para ahorrar memoria
model = YOLO(ruta_modelo).to("cpu")

# Diccionario de imágenes asociadas a cada clase
imagenes = {
    0: {"producto": "img_movil/llavero.png", "info": "img_movil/llavero_inf.png"},
    1: {"producto": "img_movil/chompa.png", "info": "img_movil/chompa_inf.png"},
    2: {"producto": "img_movil/guantes.png", "info": "img_movil/guantes_inf.png"},
    3: {"producto": "img_movil/gorro.png", "info": "img_movil/gorro_inf.png"}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deteccion', methods=['POST'])
def deteccion():
    try:
        # Obtener la imagen en base64
        data = request.json
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        
        # Decodificar la imagen en formato OpenCV
        image_np = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Redimensionar imagen para mejorar velocidad (ajusta si es necesario)
        frame = cv2.resize(frame, (640, 640))

        # Opcional: convertir a escala de grises para reducir carga de procesamiento
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Realizar la detección sin "stream=True" y sin imprimir logs
        results = model(frame, verbose=False)

        # Procesar detecciones
        detecciones = []
        for res in results:
            for caja in res.boxes:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                clase = int(caja.cls[0])

                detecciones.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2, "clase": clase,
                    "producto": imagenes.get(clase, {}).get("producto", ""),
                    "info": imagenes.get(clase, {}).get("info", "")
                })

        return jsonify({"detecciones": detecciones})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Desactiva debug en producción
