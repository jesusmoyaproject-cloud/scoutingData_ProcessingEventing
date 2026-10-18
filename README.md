# analitica_futbol ⚽📊

Pipeline de Visión Artificial, Tracking Cinemático, Clasificación de Equipos y Analítica Táctica para Fútbol.
Diseñado para alto rendimiento y despliegue híbrido tanto en **entornos locales** (Laptop / Workstation con CPU Intel y OpenVINO) como en **entornos de nube** (Google Colab / Kaggle con GPU Tesla T4 y PyTorch CUDA).

---

## 🏗️ Arquitectura del Sistema

```
analitica_futbol/
│
├── configs/                           # Archivos de configuración YAML
│   ├── base.yaml                      # Parámetros constantes (cancha FIFA, clases, umbrales)
│   ├── inference_local.yaml           # Configuración para laptop (OpenVINO / CPU)
│   ├── inference_cloud.yaml           # Configuración para Colab/Kaggle (PyTorch / CUDA)
│   └── botsort_sports.yaml            # Parámetros ajustados de BoT-SORT
│
├── data/                              # Almacenamiento de datos (ignorado en Git)
│   ├── raw/                           # Videos originales .mp4
│   └── processed/                     # Salidas de tracking (.json), eventos (.csv) y base de datos (.sqlite)
│
├── models/                            # Pesos y redes neuronales optimizadas
│   ├── modelPlayer.pt                 # Detección de jugadores (PyTorch CUDA)
│   ├── modelBall.pt                   # Detección de balón (PyTorch CUDA)
│   ├── modelKeypoint.pt               # Detección de 30 keypoints (PyTorch CUDA)
│   ├── modelPlayer_openvino_model/    # Modelo optimizado Intel OpenVINO IR
│   ├── modelBall_openvino_model/      # Modelo optimizado Intel OpenVINO IR
│   └── modelKeypoint_openvino_model/  # Modelo optimizado Intel OpenVINO IR
│
├── src/                               # Código fuente principal
│   ├── perception/                    # Módulo 1: Visión Artificial
│   │   ├── detector.py                # Inferencia YOLO unificada (PyTorch o OpenVINO)
│   │   └── homography.py              # Homografía USAC_MAGSAC + Tracker Cinemático + Auditoría
│   │
│   ├── tracking/                      # Módulo 2: Seguimiento de trayectorias
│   │   └── botsort.py                 # BoT-SORT, PlayerKinematicTracker y BallTracker
│   │
│   ├── appearance/                    # Módulo 3: Apariencia y Agrupamiento
│   │   ├── color_extractor.py         # Extracción de color de camiseta (ROI Torso + HSV + CIELAB)
│   │   └── clustering.py              # Agrupamiento no supervisado K-Means (Equipos, Porteros, Árbitros)
│   │
│   ├── analytics/                     # Módulo 4: Lógica de Negocio y Eventos
│   │   ├── ball_possession.py         # Control de posesión y porcentajes de equipo
│   │   └── events.py                  # Detección de pases, recuperaciones y remates
│   │
│   ├── storage/                       # Módulo 5: Persistencia de Datos
│   │   └── exporter.py                # Guardado de contratos JSON, CSVs tácticos y SQLite
│   │
│   ├── visualization/                 # Módulo 6: Renderizado de salidas
│   │   └── renderer.py                # Overlays de cámara, radar minimapa 2D y composición
│   │
│   └── utils/                         # Utilidades transversales
│       ├── env.py                     # Detector automático de entorno (Local vs Colab/Kaggle)
│       ├── geometry.py                # Proyecciones perspectiva y coordenadas de cancha FIFA
│       ├── video_stream.py            # Lectura multihilo asíncrona (ThreadedVideoReader) y AsyncVideoWriter
│       └── config_loader.py           # Cargador y fusionador jerárquico de configuraciones YAML
│
├── tests/                             # Pruebas unitarias
│   ├── test_tracker.py                # Pruebas de estabilización e interpolación cinemática
│   ├── test_color_clustering.py       # Pruebas de extracción de color y K-Means
│   └── test_analytics.py              # Pruebas de posesión y eventos tácticos
│
├── main.py                            # Orquestador Principal (End-to-End)
├── requirements.txt                   # Dependencias del proyecto
└── README.md                          # Documentación del proyecto y guía de instalación
```

---

## ⚡ Optimizaciones Clave Preservadas

1. **Zero HTTP Overhead:** Inferencia directa en memoria compartida sin capas REST ni sockets intermedios.
2. **USAC_MAGSAC + Robustez Temporal:** Corrección de homografía con validación de deriva de centro (*Center Shift*), rotación, escala y esquinas (*Corner Shift*), eliminando el efecto de parpadeo (*jitter*).
3. **Multi-rate Player Inferencia:** Parámetro `player_stride` para ejecutar detección pesada de jugadores cada $N$ frames con interpolación lineal balística en frames intermedios.
4. **I/O Asíncrono Multihilo:**
   - `ThreadedVideoReader`: Prefetching desacoplado en background para eliminar el cuello de botella de decodificación FFmpeg/OpenCV.
   - `AsyncVideoWriter`: Escritura en hilo separado sin pausar el ciclo de inferencia.
5. **Modo Headless Ultra-Rápido:** En servidores y notebooks de la nube (Kaggle/Colab), omite renderizado de video generando directamente el contrato estructurado JSON y SQLite.

---

## 🚀 Guía de Instalación y Uso

### 1. Requisitos Previos

```bash
python -m venv .venv
source .venv/bin/activate  # En Linux / macOS
# o en Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Ejecución Local (Laptop / CPU con OpenVINO)

Procesar video con overlays visuales y radar minimapa:
```bash
python main.py --config configs/inference_local.yaml
```

Procesamiento rápido de prueba (primeros 300 frames):
```bash
python main.py --config configs/inference_local.yaml --max-frames 300
```

Modo Headless (máxima velocidad, solo genera datos):
```bash
python main.py --config configs/inference_local.yaml --headless --export-sqlite
```

### 3. Ejecución en la Nube (Google Colab / Kaggle T4 con PyTorch CUDA)

```bash
python main.py --config configs/inference_cloud.yaml --headless
```

---

## 🧪 Pruebas Unitarias

Ejecutar la suite completa de pruebas unitarias:
```bash
python -m unittest discover tests
```
