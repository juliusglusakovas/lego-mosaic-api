# mosaic_core.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np


# -----------------------------
# 1) ЖЕСТКАЯ ПАЛИТРА 5 ТОНов
# -----------------------------
# Порядок фиксируем и не меняем
# LEGO palette - normalized to user's specification
LEGO_PALETTE = np.array([
    [220, 224, 225],  # almost white (318)
    [150, 160, 171],  # light gray (485)
    [88,  99,  110],  # mid gray (506)
    [35,  46,  59],   # dark gray (864)
    [20,  25,  30],   # almost black (966)
], dtype=np.uint8)

# Keep backward compatibility
PALETTE_5: List[Tuple[int, int, int]] = [
    (220, 224, 225),  # 318 — почти белый
    (150, 160, 171),  # 485 — светло-серый
    (88,  99,  110),  # 506 — средний серый
    (35,  46,  59),   # 864 — тёмно-серый
    (20,  25,  30),   # 966 — почти чёрный
]

PALETTE_5_ARR = np.array(PALETTE_5, dtype=np.float32)  # (5,3)
MAX_PER_COLOR = 1160  # competitor inventory: не более 1160 «плиток» на цвет

# -----------------------------
# Edge enhancement parameters (tunable)
# -----------------------------
# Unsharp mask strength: alpha controls how much sharpening is applied
ENHANCE_UNSHARP_ALPHA = 0.7  # 0.0 = no sharpening, 1.0 = full sharpening
ENHANCE_UNSHARP_SIGMA = 1.0  # Gaussian blur radius for unsharp mask

# Contrast enhancement: c > 1.0 increases contrast around midtones
ENHANCE_CONTRAST_FACTOR = 1.15  # 1.0 = no change, >1.0 = more contrast

# Gamma adjustment: < 1.0 brightens, > 1.0 darkens, preserves midtones
ENHANCE_GAMMA = 0.95  # Slightly brighten while preserving structure


def rgb_to_luminance(arr: np.ndarray) -> np.ndarray:
    """Convert RGB array (any shape ending with 3) to luminance in [0,1]."""
    arr_f = arr.astype(np.float32) / 255.0
    r = arr_f[..., 0]
    g = arr_f[..., 1]
    b = arr_f[..., 2]
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.clip(Y, 0.0, 1.0)


PALETTE_LUM = rgb_to_luminance(PALETTE_5_ARR)


def resize_lanczos(im: Image.Image, size: int) -> Image.Image:
    """Resize to size×size with LANCZOS."""
    return im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def to_64_lanczos(im: Image.Image) -> Image.Image:
    """Уменьшаем исходник до 64×64 методом LANCZOS (обёртка для совместимости)."""
    return resize_lanczos(im, 64)


def nearest_palette_indices(rgb: np.ndarray) -> np.ndarray:
    """rgb: (H,W,3) float32[0..255] -> idx(H,W) по ближайшему цвету из 5."""
    diff = rgb[:, :, None, :] - PALETTE_5_ARR[None, None, :, :]
    dist2 = (diff * diff).sum(axis=-1)
    idx = dist2.argmin(axis=-1).astype(np.uint8)
    return idx


def quantize_to_5(im_64_rgb: Image.Image) -> Image.Image:
    """Квантование 64×64 в строго 5 цветов по ближайшему (без дезеринга)."""
    arr = np.asarray(im_64_rgb, dtype=np.uint8)
    idx = nearest_palette_indices(arr.astype(np.float32))
    out = PALETTE_5_ARR[idx].astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def idx_image(im_64_rgb: Image.Image) -> np.ndarray:
    """Преобразовать 64×64 RGB в индексы 0..4 (по ближайшему цвету)."""
    arr = np.asarray(im_64_rgb, dtype=np.uint8)
    idx = nearest_palette_indices(arr.astype(np.float32))
    return idx


def accuracy_vs_ref(im_64_rgb: Image.Image, ref_64_rgb: Image.Image) -> float:
    """Доля совпавших пикселей по индексам палитры."""
    a = idx_image(im_64_rgb)
    b = idx_image(ref_64_rgb)
    return float((a == b).mean())


def color_stats(im_64_rgb: Image.Image) -> Dict[str, float]:
    """Проценты 5 индексов на выходе."""
    idx = idx_image(im_64_rgb).ravel()
    counts = np.bincount(idx, minlength=5)
    share = counts / counts.sum()
    return {f"{i}": float(share[i]) for i in range(5)}


def count_by_color(idx: np.ndarray) -> np.ndarray:
    """Вернуть абсолютные количества по каждому цвету."""
    return np.bincount(idx.reshape(-1), minlength=5)


def preview_x5(im_64_rgb: Image.Image) -> Image.Image:
    """NEAREST-превью ×5 (320×320) без блюра."""
    return im_64_rgb.resize((320, 320), Image.Resampling.NEAREST)


def enhance_faces_and_edges(img: Image.Image) -> Image.Image:
    """
    Light edge-aware enhancement before quantization.
    
    Goals:
    - Slightly sharpen edges (eyes, mouth, outline).
    - Slightly increase midtone contrast on face.
    - Keep background noise soft, no big blocks or artefacts.
    - Fully deterministic, pure-PIL / numpy, no external APIs.
    
    This function enhances contours and facial structure to make them more readable
    and "collected" in the final 5-color mosaic, while keeping background smooth.
    
    Tuning parameters (module-level constants):
    - ENHANCE_UNSHARP_ALPHA: sharpening strength (default 0.7)
    - ENHANCE_UNSHARP_SIGMA: blur radius for unsharp (default 1.0)
    - ENHANCE_CONTRAST_FACTOR: contrast boost (default 1.15)
    - ENHANCE_GAMMA: brightness adjustment (default 0.95)
    """
    # Convert to numpy uint8 [0, 255] - rgb_to_luminance expects [0..255] range
    arr = np.asarray(img, dtype=np.uint8)
    h, w, c = arr.shape
    
    # Convert to grayscale for processing (work on luminance)
    # rgb_to_luminance() internally divides by 255, so it expects [0..255] range
    if c == 3:
        lum = rgb_to_luminance(arr)  # Returns [0, 1] range
    else:
        # For grayscale, normalize manually
        lum = arr[:, :, 0].astype(np.float32) / 255.0 if arr.ndim == 3 else arr.astype(np.float32) / 255.0
    
    # Step 1: Light unsharp mask to sharpen edges
    # Create blurred version
    lum_img = Image.fromarray((lum * 255.0).astype(np.uint8), mode="L")
    blurred = lum_img.filter(ImageFilter.GaussianBlur(radius=ENHANCE_UNSHARP_SIGMA))
    blurred_arr = np.asarray(blurred, dtype=np.float32) / 255.0
    
    # Unsharp: orig + alpha * (orig - blurred)
    sharpened = lum + ENHANCE_UNSHARP_ALPHA * (lum - blurred_arr)
    sharpened = np.clip(sharpened, 0.0, 1.0)
    
    # Step 2: Slight contrast increase around midtones
    # enh = (sharpened - m) * c + m, where m = 0.5
    m = 0.5
    enhanced = (sharpened - m) * ENHANCE_CONTRAST_FACTOR + m
    enhanced = np.clip(enhanced, 0.0, 1.0)
    
    # Step 3: Soft gamma adjustment to enhance light/dark while preserving midtones
    enhanced = np.clip(enhanced, 0.0, 1.0) ** ENHANCE_GAMMA
    
    # Convert back to RGB (same luminance for all channels)
    if c == 3:
        enhanced_rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)
    else:
        enhanced_rgb = enhanced
    
    # Convert back to uint8 [0, 255] and return Image
    result_arr = (enhanced_rgb * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(result_arr, mode="RGB")


def quantize_with_inventory(
    im_64_rgb: Image.Image,
    max_per_color: int = MAX_PER_COLOR,
) -> Image.Image:
    """
    Inventory-constrained, edge-aware luminance quantisation.
    
    Quantises 64×64 RGB into PALETTE_5 respecting MAX_PER_COLOR = 1160 per color.
    Uses importance map (gradient magnitude) to preserve high-detail regions
    (faces, edges) and preferentially reassign overflow pixels from flat background.
    
    Algorithm:
      1) Ideal luminance-based assignment (min squared error per pixel).
      2) Compute importance map from luminance gradient (edges/details = high importance).
      3) If any color exceeds inventory, reassign overflowing pixels starting from
         LOW-importance pixels first (flat background), preserving HIGH-importance
         pixels (edges, facial structure) as long as possible.
      4) Reassignment uses next-best colors by error, respecting capacity limits.
    
    This produces cleaner faces and edges while background carries most dithering noise,
    matching reference mosaic quality better than random redistribution.
    
    Tuning importance weighting:
    - Current: importance = gradient_magnitude (normalized to [0,1])
    - To emphasize edges more: multiply importance by factor > 1.0 before normalization
    - To protect center more: add distance-from-border weight (e.g., center_weight = 1.0 + 0.3 * (1 - distance_to_center / max_distance))
    - To flatten importance: use importance = np.ones_like(grad) * 0.5 (uniform, falls back to error-based ordering)
    """
    arr = np.asarray(im_64_rgb, dtype=np.uint8)
    h, w, _ = arr.shape

    # Phase 1: Ideal assignment based on luminance
    lum = rgb_to_luminance(arr)
    lum_flat = lum.reshape(-1, 1)
    errors = (lum_flat - PALETTE_LUM[None, :]) ** 2  # (N,5)
    sorted_colors = np.argsort(errors, axis=1)
    idx_flat = sorted_colors[:, 0].astype(np.int16)
    counts = np.bincount(idx_flat, minlength=5).astype(np.int32)

    overflow = np.maximum(0, counts - max_per_color)
    capacity = np.maximum(0, max_per_color - counts)

    # Early exit if no overflow (fast path)
    if not overflow.any():
        out = PALETTE_5_ARR[idx_flat.reshape(h, w)].astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    # Phase 2: Compute importance map from luminance gradient
    # Higher gradient = edges/details = more important to preserve
    gy, gx = np.gradient(lum)
    grad = np.sqrt(gx * gx + gy * gy)  # (H,W) gradient magnitude
    
    # Enhance strong gradients: apply non-linearity to boost edge importance
    # grad^0.7 raises weak edges but preserves strong ones, making face contours more prominent
    grad_enhanced = grad ** 0.7
    
    # Normalize to [0, 1] for consistent importance scaling
    grad_max = grad_enhanced.max()
    if grad_max > 1e-6:
        importance = grad_enhanced / grad_max
    else:
        # Uniform image (no edges), use uniform importance
        importance = np.ones_like(grad) * 0.5
    
    # Dilate importance map to "thicken" protection zones around edges
    # This makes the protection zone wider around face contours, eyes, mouth
    # Simple 3×3 max filter (dilation) applied 1-2 times
    importance_dilated = importance.copy()
    for _ in range(2):  # 2 passes for wider protection zone
        # 3×3 max filter: each pixel = max of itself and 8 neighbors
        padded = np.pad(importance_dilated, ((1, 1), (1, 1)), mode='edge')
        for di in range(3):
            for dj in range(3):
                if di == 1 and dj == 1:
                    continue  # Skip center (already in importance_dilated)
                window = padded[di:di+h, dj:dj+w]
                importance_dilated = np.maximum(importance_dilated, window)
    
    importance = importance_dilated
    importance = np.clip(importance, 0.0, 1.0)  # Ensure [0, 1] range
    
    importance_flat = importance.reshape(-1)  # (N,) same length as idx_flat

    # Phase 3: Importance-aware overflow fix with background smoothing
    # Split pixels into important and background sets to avoid patchiness
    importance_median = np.median(importance_flat)
    
    # Deterministic RNG for background shuffling (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    
    # Phase 3: Process each overflowing color
    for color in range(5):
        extra = int(overflow[color])
        if extra <= 0:
            continue
        
        candidates = np.flatnonzero(idx_flat == color)
        if candidates.size == 0:
            continue
        
        # (A) FACE PROTECTION: Split candidates into important and background
        cand_imp = importance_flat[candidates]
        important_mask = cand_imp > importance_median
        background_mask = ~important_mask
        
        important_pixels = candidates[important_mask]
        background_pixels = candidates[background_mask]
        
        # (B) BACKGROUND SMOOTHING: Shuffle background pixels deterministically
        # This breaks up patches and creates even TV-noise look
        if background_pixels.size > 0:
            background_pixels_shuffled = background_pixels.copy()
            rng.shuffle(background_pixels_shuffled)
        else:
            background_pixels_shuffled = background_pixels
        
        # (A) FACE PROTECTION: Sort important pixels deterministically (DESCENDING importance)
        # Most important pixels (strongest edges) are processed LAST to maximize protection
        if important_pixels.size > 0:
            imp_imp = importance_flat[important_pixels]
            # Deterministic sort: descending importance (highest first in sort, but we reverse)
            # Use pixel index as secondary key for tie-breaking
            important_order = np.lexsort((important_pixels, -imp_imp))  # Negative for descending
            important_pixels_sorted = important_pixels[important_order]
        else:
            important_pixels_sorted = important_pixels
        
        # Process background first, then important (face protection)
        candidates_ordered = np.concatenate([background_pixels_shuffled, important_pixels_sorted])

        # Reassign candidates: background first, then important
        for pos in candidates_ordered:
            if overflow[color] <= 0:
                break
            
            current_color = idx_flat[pos]
            
            # (C) NEIGHBOR-COLOR REPLACEMENT: Only use colors close in brightness
            # Palette is ordered by brightness: 0=lightest, 4=darkest
            # Prefer palette_idx ± 1, then ± 2, to avoid sharp artifacts
            preferred_deltas = [1, -1, 2, -2, 3, -3, 4, -4]  # ±1, ±2, ±3, ±4
            
            found_alt = False
            for delta in preferred_deltas:
                alt = current_color + delta
                if alt < 0 or alt >= 5:
                    continue
                if alt == color:
                    continue
                if capacity[alt] <= 0:
                    continue
                
                # Reassign to neighbor color (close in brightness)
                idx_flat[pos] = alt
                overflow[color] -= 1
                capacity[alt] -= 1
                found_alt = True
                break
            
            # Fallback: if no neighbor color available, use best alternative from sorted_colors
            if not found_alt:
                for alt in sorted_colors[pos]:
                    if alt == color:
                        continue
                    if capacity[alt] <= 0:
                        continue
                    idx_flat[pos] = alt
                    overflow[color] -= 1
                    capacity[alt] -= 1
                    break

    # Final validation (debug only, commented for production)
    # final_counts = np.bincount(idx_flat, minlength=5).astype(np.int32)
    # assert np.all(final_counts <= MAX_PER_COLOR), f"Inventory violation: {final_counts}"

    out = PALETTE_5_ARR[idx_flat.reshape(h, w)].astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


# -----------------------------
# 2) ПАРАМЕТРИЧЕСКИЙ ПРЕСЕТ
# -----------------------------
@dataclass
class Preset:
    blur: float = 0.0          # sigma для GaussianBlur
    unsharp_radius: float = 0.0
    unsharp_percent: int = 0   # 0..500
    contrast: float = 1.0
    brightness: float = 1.0
    gamma: float = 1.0


def apply_preset_once(
    og: Image.Image,
    p: Preset,
    size: int = 64,
    max_per_color: int = MAX_PER_COLOR,
) -> Image.Image:
    """
    Прогоняем OG через пресет и квантование в 5 цветов.
    
    Pipeline:
    1) Resize to size×size (LANCZOS)
    2) Apply preset filters (blur, unsharp, contrast, brightness, gamma)
    3) Edge-aware enhancement (enhance_faces_and_edges) - strengthens contours and faces
    4) Inventory-constrained quantization (quantize_with_inventory)
    """
    im = resize_lanczos(og, size)

    if p.blur > 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=p.blur))

    if p.unsharp_radius > 0 and p.unsharp_percent > 0:
        im = im.filter(ImageFilter.UnsharpMask(
            radius=p.unsharp_radius,
            percent=int(p.unsharp_percent),
            threshold=0
        ))

    if p.contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(p.contrast)

    if p.brightness != 1.0:
        im = ImageEnhance.Brightness(im).enhance(p.brightness)

    if p.gamma != 1.0:
        arr = np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0
        arr = np.clip(arr ** (1.0 / max(p.gamma, 1e-6)), 0.0, 1.0)
        im = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    # Edge-aware enhancement: strengthens contours and faces before quantization
    # This makes faces more readable and "collected" in the final mosaic
    im = enhance_faces_and_edges(im)

    im = quantize_with_inventory(im, max_per_color=max_per_color)
    return im


def save_png_64(path: str, im_64_rgb: Image.Image) -> None:
    if im_64_rgb.size != (64, 64):
        im_64_rgb = im_64_rgb.resize((64, 64), Image.Resampling.NEAREST)
    im_64_rgb.save(path, format="PNG")


def evaluate_preset_multi(
    preset: Preset,
    og_list: List[Image.Image],
    ref_list: List[Image.Image],
) -> Tuple[float, List[float], List[Image.Image]]:
    """
    Оценить пресет на нескольких парах OG+REF.
    
    Args:
        preset: Пресет для применения
        og_list: Список оригинальных изображений
        ref_list: Список референсов (64×64), должен совпадать по длине с og_list
    
    Returns:
        (mean_accuracy, per_pair_accuracy, output_images):
        - mean_accuracy: средняя точность по всем парам
        - per_pair_accuracy: список точностей для каждой пары
        - output_images: список мозаик (64×64) для каждой пары
    """
    assert len(og_list) == len(ref_list), "OG and REF lists must have same length"
    
    per_pair_accuracy: List[float] = []
    output_images: List[Image.Image] = []
    
    for og, ref in zip(og_list, ref_list):
        mosaic = apply_preset_once(og, preset)
        output_images.append(mosaic)
        
        # Подготовить REF к 64×64 если нужно
        ref_64 = ref.convert("RGB")
        if ref_64.size != (64, 64):
            ref_64 = ref_64.resize((64, 64), Image.Resampling.NEAREST)
        
        acc = accuracy_vs_ref(mosaic, ref_64)
        per_pair_accuracy.append(acc)
    
    mean_accuracy = sum(per_pair_accuracy) / len(per_pair_accuracy) if per_pair_accuracy else 0.0
    
    return mean_accuracy, per_pair_accuracy, output_images


def aggregate_color_counts(mosaics: List[Image.Image]) -> np.ndarray:
    """
    Суммировать счётчики цветов по всем мозаикам.
    
    Args:
        mosaics: Список мозаик 64×64
    
    Returns:
        Массив из 5 элементов: суммарные количества каждого цвета
    """
    total_counts = np.zeros(5, dtype=np.int32)
    for mosaic in mosaics:
        idx = idx_image(mosaic)
        counts = count_by_color(idx)
        total_counts += counts
    return total_counts


def render_3d_mosaic(pixel_mosaic: np.ndarray, tile_images: Dict[int, Image.Image], max_output_size: Tuple[int, int] = (1920, 1080)) -> Image.Image:
    """
    Render a 3D mosaic by replacing each pixel with a corresponding 3D LEGO tile image.
    The output will be scaled to fit within max_output_size (default: Full HD 1920×1080).
    
    Args:
        pixel_mosaic: 2D numpy array of shape (H, W) with color indices (0-4)
        tile_images: Dictionary mapping color index (0-4) to PIL Image of the tile
        max_output_size: Maximum output size (width, height) in pixels. Default: (1920, 1080) for Full HD.
                        The image will be scaled proportionally to fit within these dimensions.
    
    Returns:
        PIL Image of the stitched 3D mosaic, scaled to fit within max_output_size
    """
    h, w = pixel_mosaic.shape
    
    if not tile_images:
        raise ValueError("tile_images dictionary is empty")
    
    max_w, max_h = max_output_size
    
    # Calculate optimal tile size to fit within max_output_size
    # For square mosaics, we fit into the smaller dimension (usually height for Full HD)
    # We want: w * tile_size <= max_w and h * tile_size <= max_h
    tile_size_w = max_w // w
    tile_size_h = max_h // h
    tile_size = min(tile_size_w, tile_size_h)
    
    # Ensure minimum tile size for quality (at least 8 pixels)
    tile_size = max(8, tile_size)
    
    # For square mosaics, ensure we fit within the smaller dimension
    # (Full HD is 1920×1080, so square should be max 1080×1080)
    if w == h:  # Square mosaic
        max_square_dim = min(max_w, max_h)
        tile_size = min(tile_size, max_square_dim // w)
    
    # Scale tiles to calculated size
    scaled_tiles: Dict[int, Image.Image] = {}
    for idx, tile in tile_images.items():
        if tile.size[0] != tile_size or tile.size[1] != tile_size:
            scaled_tiles[idx] = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        else:
            scaled_tiles[idx] = tile
    
    # Create output image at calculated size
    output_w = w * tile_size
    output_h = h * tile_size
    output_img = Image.new("RGB", (output_w, output_h))
    
    # Paste each tile
    for y in range(h):
        for x in range(w):
            color_idx = int(pixel_mosaic[y, x])
            if color_idx not in scaled_tiles:
                raise ValueError(f"Color index {color_idx} not found in tile_images")
            
            tile = scaled_tiles[color_idx]
            
            # Calculate paste position
            paste_x = x * tile_size
            paste_y = y * tile_size
            
            # Paste tile onto output
            output_img.paste(tile, (paste_x, paste_y))
    
    # Scale final image to fit exactly within max_output_size if needed
    # (maintain aspect ratio)
    if output_w > max_w or output_h > max_h:
        # Calculate scaling factor to fit within max dimensions
        scale_w = max_w / output_w
        scale_h = max_h / output_h
        scale = min(scale_w, scale_h)
        
        new_w = int(output_w * scale)
        new_h = int(output_h * scale)
        output_img = output_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return output_img


def debug_quantize_solid_colors(output_dir: str = "debug_quantize") -> None:
    """
    Helper for synthetic tests. Generates several patterns, runs quantisation
    and prints per-color counts (must stay <= MAX_PER_COLOR).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tests: Dict[str, Image.Image] = {}
    for val in (32, 96, 160, 224):
        arr = np.full((64, 64, 3), val, dtype=np.uint8)
        tests[f"solid_{val}"] = Image.fromarray(arr, mode="RGB")

    gradient = np.linspace(0, 255, 64, dtype=np.uint8)
    grad_arr = np.tile(gradient, (64, 1))
    tests["gradient_horizontal"] = Image.fromarray(
        np.stack([grad_arr] * 3, axis=-1), mode="RGB"
    )

    stripes = np.zeros((64, 64, 3), dtype=np.uint8)
    stripes[:, ::2] = 210
    stripes[:, 1::2] = 30
    tests["stripes"] = Image.fromarray(stripes, mode="RGB")

    for name, img in tests.items():
        quant = quantize_with_inventory(img)
        idx = idx_image(quant)
        counts = np.bincount(idx.reshape(-1), minlength=5)
        quant.save(out_dir / f"{name}.png")
        print(f"{name}: counts={counts.tolist()}")



