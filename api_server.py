from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from PIL import Image

from mosaic_core import (
    Preset,
    apply_preset_once,
    idx_image,
    render_3d_mosaic,
    MAX_PER_COLOR,
)

# ------------------------------------------------------
# НАСТРОЙКИ
# ------------------------------------------------------

# Путь к твоему лучшему пресету (имя файла должно совпадать с тем, что в репо)
PRESET_PATH = Path("lego_mosaic_best_preset (4).json")

# Маппинг индекса палитры -> имя PNG-файла тайла
# Это те же имена, что и в твоём app.py
TILE_MAP = {
    0: "RGB (220, 224, 225).png",
    1: "RGB (150, 160, 171).png",
    2: "RGB (88, 99, 110).png",
    3: "RGB (35, 46, 59).png",
    4: "RGB (20, 25, 30).png",
}

_tile_cache: Optional[Dict[int, Image.Image]] = None
_global_preset: Optional[Preset] = None

app = FastAPI(
    title="LEGO Mosaic 3D API",
    description="Генерирует 3D LEGO-мозаику по фото, используя заранее подобранный пресет и тот же алгоритм, что в Gradio.",
    version="1.0.0",
)

# Разрешим CORS (чтобы можно было дергать API из любого фронта)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ------------------------------------------------------


def load_preset_from_json(path: Path) -> Preset:
    """
    Загрузить Preset из JSON.
    Логика та же, что в твоём app.py: поля blur, unsharp_radius, unsharp_percent,
    contrast, brightness, gamma. :contentReference[oaicite:5]{index=5}
    """
    if not path.exists():
        raise FileNotFoundError(f"Preset JSON not found at: {path}")
    with path.open("r", encoding="utf-8") as f:
        preset_dict = json.load(f)

    return Preset(
        blur=float(preset_dict.get("blur", 0.0)),
        unsharp_radius=float(preset_dict.get("unsharp_radius", 0.0)),
        unsharp_percent=int(preset_dict.get("unsharp_percent", 0)),
        contrast=float(preset_dict.get("contrast", 1.0)),
        brightness=float(preset_dict.get("brightness", 1.0)),
        gamma=float(preset_dict.get("gamma", 1.0)),
    )


def load_tile_images(tiles_dir: str = "tiles") -> Dict[int, Image.Image]:
    """
    Загрузить PNG-тайлы для 3D-мозаики.

    Логика поиска совпадает с твоим load_tile_images в app.py:
    - сначала смотрим в tiles/
    - затем в корне проекта
    - затем в /mnt/data
    - затем рядом с файлом. :contentReference[oaicite:6]{index=6}
    """
    global _tile_cache

    if _tile_cache is not None:
        return _tile_cache

    from pathlib import Path as P

    search_paths = [
        P(tiles_dir),                  # User-specified directory
        P("tiles"),                    # Default tiles directory
        P("/mnt/data"),                # Cursor / контейнеры
        P("."),                        # Project root
        P(__file__).parent / "tiles",  # рядом с api_server.py
    ]

    tiles_path = None
    for search_path in search_paths:
        if search_path.exists():
            test_file = search_path / TILE_MAP[0]
            if test_file.exists():
                tiles_path = search_path
                break

    if tiles_path is None:
        # fallback: try прям в корне
        for filename in TILE_MAP.values():
            if P(filename).exists():
                tiles_path = P(".")
                break

    if tiles_path is None:
        raise FileNotFoundError(
            "Tile images not found. "
            "Please put the following PNG files in 'tiles/' or project root:\n"
            + "\n".join([f"  - {fname}" for fname in TILE_MAP.values()])
        )

    tile_images: Dict[int, Image.Image] = {}

    for idx, filename in TILE_MAP.items():
        tile_path = tiles_path / filename
        if not tile_path.exists():
            raise FileNotFoundError(
                f"Tile image not found: {filename} in {tiles_path}. "
                f"Please ensure all tile images are present."
            )

        try:
            tile_img = Image.open(tile_path).convert("RGB")
            tile_images[idx] = tile_img
        except Exception as e:
            raise IOError(f"Failed to load tile image {filename}: {e}")

    _tile_cache = tile_images
    return tile_images


# ------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ ПРИ СТАРТЕ (как бы "preload")
# ------------------------------------------------------


@app.on_event("startup")
async def startup_event():
    global _global_preset, _tile_cache
    # Загружаем фиксированный пресет
    _global_preset = load_preset_from_json(PRESET_PATH)
    # Заранее грузим тайлы (если не получится — упадём сразу при старте)
    _tile_cache = load_tile_images()


# ------------------------------------------------------
# ЭНДПОИНТЫ
# ------------------------------------------------------


@app.get("/health")
async def health():
    """Проверка живости сервиса."""
    return {"status": "ok"}


@app.post("/mosaic3d")
async def generate_lego_mosaic_3d(
    file: UploadFile = File(..., description="Фото для мозаики"),
    size: int = Form(64, description="Размер мозаики: 64 или 96"),
):
    """
    Сгенерировать 3D LEGO-мозаику.

    Вход (multipart/form-data):
      - file: изображение (jpeg/png/…)
      - size: 64 или 96

    Алгоритм:
      1) apply_preset_once(og, preset, size, max_per_color)
         - resize (LANCZOS)
         - blur / unsharp / contrast / brightness / gamma
         - enhance_faces_and_edges
         - quantize_with_inventory (с ограничением MAX_PER_COLOR / 2900) :contentReference[oaicite:7]{index=7}
      2) idx_image(...)
      3) render_3d_mosaic(...)

    Выход:
      - image/png: готовая 3D-мозаика.
    """
    global _global_preset

    if _global_preset is None:
        raise HTTPException(status_code=500, detail="Preset not loaded")

    if size not in (64, 96):
        raise HTTPException(status_code=400, detail="size must be 64 or 96")

    # Читаем входное изображение
    try:
        raw_bytes = await file.read()
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    # max_per_color — ровно как в твоём generate_mosaic_from_preset :contentReference[oaicite:8]{index=8}
    if size == 96:
        max_per_color = 2900
    else:
        max_per_color = MAX_PER_COLOR

    # 1) 2D пиксельная мозаика той же функцией, что и в Gradio
    try:
        mosaic_2d = apply_preset_once(
            img,
            _global_preset,
            size=size,
            max_per_color=max_per_color,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating 2D mosaic: {e}")

    # 2) Индексы 0..4 (та же idx_image из mosaic_core)
    try:
        indices = idx_image(mosaic_2d)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing palette indices: {e}")

    # 3) 3D-рэндер тайлами (та же функция render_3d_mosaic) :contentReference[oaicite:9]{index=9}
    try:
        tile_images = load_tile_images()
        mosaic_3d = render_3d_mosaic(
            indices,
            tile_images=tile_images,
            max_output_size=(1920, 1080),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating 3D mosaic: {e}")

    # 4) Отдаём PNG
    buf = BytesIO()
    mosaic_3d.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
