from pathlib import Path
import cv2
import pandas as pd
from PIL import Image
from ultralytics import YOLO

ANIMAL_CLASSES = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Color AZUL en formato BGR (OpenCV usa BGR)
COLOR_AZUL_BGR = (255, 0, 0)
COLOR_TEXTO_BGR = (255, 255, 255)


def dibujar_rectangulos_azules(img_path: Path, cajas, model_names: dict) -> Image.Image:
    """Dibuja un rectángulo azul y etiqueta sobre cada animal detectado."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen con OpenCV: {img_path}")

    alto, ancho, _ = img_bgr.shape
    grosor = max(2, int(round(min(ancho, alto) / 250)))
    escala_fuente = max(0.5, min(ancho, alto) / 1000)
    grosor_texto = max(1, int(round(escala_fuente * 2)))

    for box in cajas:
        x1, y1, x2, y2 = [int(round(coord)) for coord in box.xyxy[0].tolist()]
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        nombre_animal = ANIMAL_CLASSES.get(cls_id, model_names[cls_id])
        etiqueta = f"{nombre_animal} {conf:.2f}"

        # Rectángulo azul
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), COLOR_AZUL_BGR, grosor)

        # Fondo azul para la etiqueta
        (tw, th), baseline = cv2.getTextSize(
            etiqueta, cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, grosor_texto
        )
        top_y = max(0, y1 - th - baseline - 6)
        cv2.rectangle(
            img_bgr,
            (x1, top_y),
            (min(ancho, x1 + tw + 8), y1),
            COLOR_AZUL_BGR,
            -1,
        )
        cv2.putText(
            img_bgr,
            etiqueta,
            (x1 + 4, max(th + 2, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala_fuente,
            COLOR_TEXTO_BGR,
            grosor_texto,
            cv2.LINE_AA,
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def detectar_animales_en_carpeta(
    carpeta_entrada: str,
    carpeta_salida: str = "resultados_animales",
    modelo_peso: str = "yolo11n.pt",
    confianza_minima: float = 0.4,
    guardar_imagenes: bool = True,
) -> pd.DataFrame:
    ruta_entrada = Path(carpeta_entrada)
    ruta_salida = Path(carpeta_salida)
    ruta_imgs_anotadas = ruta_salida / "imagenes_detectadas"

    if not ruta_entrada.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {ruta_entrada}")

    ruta_salida.mkdir(parents=True, exist_ok=True)
    if guardar_imagenes:
        ruta_imgs_anotadas.mkdir(parents=True, exist_ok=True)

    model = YOLO(modelo_peso)

    imagenes = [
        p for p in ruta_entrada.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONES_VALIDAS
    ]

    if not imagenes:
        print("no hay animales")
        return pd.DataFrame()

    registros = []

    for img_path in imagenes:
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            print(f"[AVISO] Imagen corrupta omitida ({img_path.name}): {e}")
            continue

        resultados = model.predict(
            source=str(img_path),
            classes=list(ANIMAL_CLASSES.keys()),
            conf=confianza_minima,
            verbose=False,
        )

        cajas = resultados[0].boxes

        if cajas is not None and len(cajas) > 0:
            if guardar_imagenes:
                img_pil = dibujar_rectangulos_azules(img_path, cajas, model.names)
                img_pil.save(ruta_imgs_anotadas / f"det_{img_path.name}")

            for box in cajas:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [round(coord, 1) for coord in box.xyxy[0].tolist()]
                nombre_animal = ANIMAL_CLASSES.get(cls_id, model.names[cls_id])

                print(f"{img_path.name}: {nombre_animal} ({conf:.2f}) -> rectángulo azul")

                registros.append({
                    "archivo": img_path.name,
                    "ruta_original": str(img_path),
                    "animal": nombre_animal,
                    "confianza": round(conf, 4),
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                })
        else:
            print(f"{img_path.name}: no hay animales")
            registros.append({
                "archivo": img_path.name,
                "ruta_original": str(img_path),
                "animal": "no hay animales",
                "confianza": None,
                "bbox_x1": None,
                "bbox_y1": None,
                "bbox_x2": None,
                "bbox_y2": None,
            })

    df = pd.DataFrame(registros)
    csv_path = ruta_salida / "reporte_detecciones.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\nReporte guardado en: {csv_path}")
    if guardar_imagenes:
        print(f"Imágenes con rectángulo azul guardadas en: {ruta_imgs_anotadas}")

    return df


if __name__ == "__main__":
    df_resultados = detectar_animales_en_carpeta(
        carpeta_entrada=".",
        carpeta_salida="./salida_animales",
        modelo_peso="yolo11n.pt",
        confianza_minima=0.4,
        guardar_imagenes=True,
    )

    if not df_resultados.empty:
        detecciones_reales = df_resultados[df_resultados["animal"] != "no hay animales"]
        if detecciones_reales.empty:
            print("no hay animales")
        else:
            print("\nResumen de animales detectados:")
            print(detecciones_reales["animal"].value_counts())