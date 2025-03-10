from flask import Flask, render_template, Response, jsonify
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)

# Directorio base
directorio_base = os.path.dirname(os.path.abspath(__file__))

# Ruta del modelo
ruta_modelo = os.path.join(directorio_base, "models_movil", "best110.pt")

if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"El modelo no se encontró en: {ruta_modelo}")

# Cargar el modelo YOLO
model = YOLO(ruta_modelo)

# Rutas de imágenes
ruta_imagenes = os.path.join(directorio_base, "static", "img_movil")
imagenes = {
    0: {"producto": "img_movil/llavero.png", "info": "img_movil/llavero_inf.png"},
    1: {"producto": "img_movil/chompa.png", "info": "img_movil/chompa_inf.png"},
    2: {"producto": "img_movil/guantes.png", "info": "img_movil/guantes_inf.png"},
    3: {"producto": "img_movil/gorro.png", "info": "img_movil/gorro_inf.png"}
}

# Variables globales para almacenar la última detección
ultima_deteccion = {"producto": "", "info": ""}

# Iniciar la cámara
cap = cv2.VideoCapture(0)

def generar_frames():
    global ultima_deteccion
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar con YOLO
        results = model(frame, stream=True, verbose=False)

        for res in results:
            cajas = res.boxes
            for caja in cajas:
                x1, y1, x2, y2 = [int(val) for val in caja.xyxy[0]]
                clase = int(caja.cls[0])

                # Dibujar detección
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Clase {clase}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Actualizar imágenes detectadas
                if clase in imagenes:
                    ultima_deteccion = imagenes[clase]

        # Convertir frame a JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', producto=ultima_deteccion["producto"], info=ultima_deteccion["info"])

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/deteccion')
def deteccion():
    return jsonify(ultima_deteccion)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
