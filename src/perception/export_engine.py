"""
Script de exportación de modelos PyTorch (.pt) a motores TensorRT (.engine) en FP16.
Guarda los archivos generados coherentemente en la carpeta models/ (junto a los .pt y OpenVINO).

Uso:
  python -m src.perception.export_engine
  python -m src.perception.export_engine --weights models/modelPlayer.pt --half
"""

import sys
import argparse
import shutil
import logging
from pathlib import Path

# Asegurar importación de src
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO
from src.utils.env import is_cuda_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_engine")


def export_model_to_engine(
    weights_path: Path,
    device: int = 0,
    half: bool = True,
    imgsz: int = 640,
    batch: int = 1,
) -> Path:
    """
    Exporta un modelo YOLO .pt a formato TensorRT .engine en FP16 y lo ubica en models/.
    """
    weights_path = Path(weights_path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Modelo .pt no encontrado: {weights_path}")

    models_dir = weights_path.parent
    logger.info(f"🚀 Iniciando exportación TensorRT: {weights_path.name}")
    logger.info(f"   Destino final: {models_dir}")
    logger.info(f"   Precisión: {'FP16 (Half)' if half else 'FP32'}")
    logger.info(f"   Dispositivo: CUDA GPU {device}")
    logger.info(f"   Resolución base: {imgsz}x{imgsz}")

    # Cargar modelo PyTorch
    model = YOLO(str(weights_path))

    # Exportar a TensorRT
    # Ultralytics compila el motor usando trtexec o tensorrt python API
    exported_file = model.export(
        format="engine",
        device=device,
        half=half,
        imgsz=imgsz,
        batch=batch,
        verbose=True,
    )

    exported_path = Path(exported_file).resolve()
    target_engine_path = models_dir / f"{weights_path.stem}.engine"

    # Si Ultralytics lo generó en otra subcarpeta o con nombre distinto, reubicarlo en models/
    if exported_path != target_engine_path and exported_path.exists():
        shutil.move(str(exported_path), str(target_engine_path))
        logger.info(f"📦 Motor reubicado en carpeta oficial: {target_engine_path}")
    else:
        target_engine_path = exported_path

    logger.info(f"✅ Exportación completada con éxito: {target_engine_path} ({target_engine_path.stat().st_size / (1024*1024):.1f} MB)")
    return target_engine_path


def export_all_models(models_dir: Path, half: bool = True, device: int = 0):
    """Exporta los 3 modelos oficiales (Player, Ball, Keypoint) si están disponibles en models/."""
    targets = [
        ("Jugadores", models_dir / "modelPlayer.pt"),
        ("Balón", models_dir / "modelBall.pt"),
        ("Keypoints", models_dir / "modelKeypoint.pt"),
    ]

    print("=" * 65)
    print("🔥 EXPORTADOR OFICIAL A TENSORRT (.engine) EN models/")
    print(f"   Directorio: {models_dir}")
    print(f"   Precisión:  {'FP16' if half else 'FP32'}")
    print("=" * 65)

    results = {}
    for name, path in targets:
        if path.exists():
            print(f"\n---> Compilando motor para {name} ({path.name})...")
            try:
                engine_path = export_model_to_engine(path, device=device, half=half)
                results[name] = str(engine_path)
            except Exception as e:
                logger.error(f"❌ Error al compilar {name}: {e}")
                results[name] = f"Error: {e}"
        else:
            logger.warning(f"⚠️ Archivo no encontrado: {path}. Saltando...")
            results[name] = "No encontrado"

    print("\n" + "=" * 65)
    print("📋 RESUMEN DE COMPILACIÓN TENSORRT:")
    for k, v in results.items():
        print(f"   - {k:10s}: {v}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Exportar modelos YOLO .pt a TensorRT .engine en carpeta models/")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Ruta a un archivo .pt específico (opcional)")
    parser.add_argument("--device", "-d", type=int, default=0, help="ID de la GPU CUDA (default: 0)")
    parser.add_argument("--half", action="store_true", default=True, help="Habilitar precisión FP16 (recomendado para L4/T4)")
    parser.add_argument("--models-dir", type=str, default="models", help="Carpeta models/ (default: models)")
    args = parser.parse_args()

    if not is_cuda_available():
        logger.error("❌ CUDA no está disponible en este entorno. TensorRT requiere una GPU NVIDIA con controladores CUDA.")
        sys.exit(1)

    models_path = Path(args.models_dir).resolve()
    if args.weights:
        export_model_to_engine(Path(args.weights), device=args.device, half=args.half)
    else:
        export_all_models(models_path, half=args.half, device=args.device)


if __name__ == "__main__":
    main()
