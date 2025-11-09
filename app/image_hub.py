"""Utility functions supporting the Image Processing Hub."""

from __future__ import annotations

import io
import math
import zipfile
import random
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"}


def extension_ok(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def session_dir(base: Path, session_id: str) -> Path:
    directory = base / "image_hub" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_originals(base: Path, session_id: str) -> list[Path]:
    folder = session_dir(base, session_id)
    originals: list[Path] = []
    for ext in ALLOWED_EXTENSIONS:
        originals.extend(folder.glob(f"*_original.{ext}"))
    return originals


def generate_image_id() -> str:
    return uuid4().hex


def save_upload(file_storage, upload_root: Path, session_id: str) -> dict[str, Any]:
    if not extension_ok(file_storage.filename or ""):
        raise ValueError("Unsupported file type.")
    image_id = generate_image_id()
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    session_path = session_dir(upload_root, session_id)
    original_path = session_path / f"{image_id}_original.{ext}"
    file_storage.save(original_path)

    # Generate thumbnail preview
    thumb_path = session_path / f"{image_id}_thumb.jpg"
    with Image.open(original_path) as img:
        img = img.convert("RGB")
        img.thumbnail((320, 320))
        img.save(thumb_path, "JPEG", quality=85)

    metadata = describe_image(original_path)
    metadata.update(
        {
            "id": image_id,
            "session_id": session_id,
            "original_path": str(original_path),
            "thumbnail_path": str(thumb_path),
        }
    )
    return metadata


def describe_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as img:
        width, height = img.size
        mode = img.mode
    size_bytes = path.stat().st_size
    return {
        "filename": path.name,
        "dimensions": f"{width} × {height}",
        "width": width,
        "height": height,
        "mode": mode,
        "size_bytes": size_bytes,
        "size": f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024 * 1024):.2f} MB",
    }


def processed_path_for(upload_root: Path, session_id: str, image_id: str) -> Path:
    return session_dir(upload_root, session_id) / f"{image_id}_processed.png"


def thumbnail_path_for(upload_root: Path, session_id: str, image_id: str) -> Path:
    return session_dir(upload_root, session_id) / f"{image_id}_thumb.jpg"


def original_glob(upload_root: Path, session_id: str, image_id: str):
    folder = session_dir(upload_root, session_id)
    for path in folder.glob(f"{image_id}_original.*"):
        return path
    return None


def create_zip(upload_root: Path, session_id: str, image_ids: list[str]) -> Path:
    session_path = session_dir(upload_root, session_id)
    zip_path = session_path / f"{session_id}_processed.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for image_id in image_ids:
            processed = processed_path_for(upload_root, session_id, image_id)
            if processed.exists():
                bundle.write(processed, arcname=processed.name)
    return zip_path


def compute_metrics(original: np.ndarray, processed: np.ndarray) -> dict[str, Any]:
    psnr_value = cv2.PSNR(original, processed) if original.size else None
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(gray_original, gray_processed, data_range=gray_processed.max() - gray_processed.min())  # type: ignore[arg-type]
    return {"psnr": round(psnr_value, 3) if psnr_value else None, "ssim": round(float(ssim_value), 4)}


@dataclass
class OperationConfig:
    resize: dict[str, Any] | None = None
    rotate: dict[str, Any] | None = None
    flip: dict[str, Any] | None = None
    brightness: dict[str, Any] | None = None
    contrast: dict[str, Any] | None = None
    blur: dict[str, Any] | None = None
    grayscale: dict[str, Any] | None = None
    clahe: dict[str, Any] | None = None


class ImageProcessor:
    """Processing pipeline for manual adjustments."""

    def __init__(self, config: dict[str, Any]):
        self.config = config or {}
        self.pipeline = [
            op
            for op in (
                "resize",
                "rotate",
                "flip",
                "brightness",
                "contrast",
                "saturation",
                "hue",
                "gamma",
                "blur",
                "denoise",
                "sharpen",
                "noise",
                "grayscale",
                "clahe",
            )
            if self._enabled(op)
        ]

    def _enabled(self, op: str) -> bool:
        block = self.config.get(op)
        if isinstance(block, dict):
            return block.get("enabled", False)
        return bool(block)

    def process(self, original_path: Path, output_path: Path) -> dict[str, Any]:
        image = cv2.imread(str(original_path))
        if image is None:
            raise ValueError(f"Unable to read image: {original_path}")
        working = image.copy()

        for op in self.pipeline:
            handler = getattr(self, f"_apply_{op}")
            working = handler(working, self.config.get(op) or {})

        cv2.imwrite(str(output_path), working)
        metrics = compute_metrics(image, working)
        return {
            "processed_path": str(output_path),
            "metrics": metrics,
        }

    def _apply_resize(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        width = int(params.get("width") or 0)
        height = int(params.get("height") or 0)
        mode = params.get("mode", "fit")
        if width <= 0 and height <= 0:
            return img
        h, w = img.shape[:2]
        if mode == "exact" and width > 0 and height > 0:
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        if width > 0 and height > 0:
            scale = min(width / w, height / h)
            width = max(1, int(w * scale))
            height = max(1, int(h * scale))
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        if width > 0:
            scale = width / w
            height = max(1, int(h * scale))
        elif height > 0:
            scale = height / h
            width = max(1, int(w * scale))
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def _apply_rotate(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        angle = float(params.get("angle") or 0)
        if angle % 360 == 0:
            return img
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += new_w / 2 - center[0]
        matrix[1, 2] += new_h / 2 - center[1]
        border = params.get("border", 0)
        return cv2.warpAffine(img, matrix, (new_w, new_h), borderValue=border)

    def _apply_flip(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        direction = params.get("direction", "horizontal")
        if direction == "horizontal":
            return cv2.flip(img, 1)
        if direction == "vertical":
            return cv2.flip(img, 0)
        if direction == "both":
            return cv2.flip(img, -1)
        return img

    def _apply_brightness(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        value = float(params.get("value") or 0)
        if value == 0:
            return img
        return cv2.convertScaleAbs(img, alpha=1.0, beta=value)

    def _apply_contrast(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        value = float(params.get("value") or 0)
        if value == 0:
            return img
        alpha = 1.0 + (value / 100.0)
        alpha = max(0.1, min(alpha, 5.0))
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    def _apply_saturation(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        value = float(params.get("value") or 0)
        if value == 0:
            return img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.int16) + value, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_hue(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        shift = int(params.get("shift") or 0) % 180
        if shift == 0:
            return img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = (h.astype(int) + shift) % 180
        hsv = cv2.merge([h.astype(np.uint8), s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_gamma(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        gamma = float(params.get("value") or 1.0)
        gamma = max(0.1, min(gamma, 5.0))
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(img, table)

    def _apply_blur(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        strength = int(params.get("strength") or 0)
        mode = params.get("mode", "gaussian")
        if strength <= 0:
            return img
        strength = max(1, min(strength, 49))
        if strength % 2 == 0:
            strength += 1
        if mode == "median":
            return cv2.medianBlur(img, strength)
        if mode == "bilateral":
            return cv2.bilateralFilter(img, strength, 75, 75)
        return cv2.GaussianBlur(img, (strength, strength), 0)

    def _apply_denoise(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        mode = params.get("mode", "fastnl")
        if mode == "median":
            ksize = int(params.get("kernel") or 5)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            return cv2.medianBlur(img, max(3, ksize))
        if mode == "bilateral":
            d = int(params.get("kernel") or 9)
            return cv2.bilateralFilter(img, d, 75, 75)
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    def _apply_sharpen(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        mode = params.get("mode", "standard")
        if mode == "strong":
            kernel = np.array(
                [
                    [-1, -1, -1, -1, -1],
                    [-1, 2, 2, 2, -1],
                    [-1, 2, 16, 2, -1],
                    [-1, 2, 2, 2, -1],
                    [-1, -1, -1, -1, -1],
                ],
                dtype=float,
            ) / 8.0
        elif mode == "soft":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        else:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    def _apply_noise(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        amount = float(params.get("amount") or 0)
        if amount <= 0:
            return img
        noise = np.random.normal(0, amount, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _apply_grayscale(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        if not params:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _apply_clahe(self, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        clip_limit = float(params.get("clip_limit") or 2.0)
        tile = int(params.get("tile_grid") or 8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class AutoDatasetBuilder:
    """Generate augmented datasets from uploaded originals."""

    def __init__(self, upload_root: Path, session_id: str):
        self.upload_root = upload_root
        self.session_id = session_id
        self.session_path = session_dir(upload_root, session_id)
        self.originals = list_originals(upload_root, session_id)

    def _compose_pipeline(self, preset: str, width: int, height: int) -> A.Compose:
        crop = A.RandomResizedCrop(
            size=(height, width),
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR,
            p=0.85,
        )
        base: list[Any] = [
            crop,
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=35, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.RandomBrightnessContrast(p=0.6),
            A.HueSaturationValue(p=0.45),
            A.GaussianBlur(blur_limit=(3, 7), p=0.25),
        ]
        if preset == "portrait":
            base += [
                A.RandomShadow(p=0.3),
                A.RandomSunFlare(src_radius=120, p=0.2),
                A.CLAHE(p=0.2),
            ]
        elif preset == "document":
            base += [
                A.MotionBlur(blur_limit=7, p=0.2),
                A.ISONoise(p=0.2),
                A.RandomRain(blur_value=3, p=0.15),
                A.Perspective(scale=(0.02, 0.08), p=0.4),
            ]
        elif preset == "creative":
            base += [
                A.ColorJitter(p=0.4),
                A.RandomFog(p=0.2),
                A.RandomSnow(p=0.2),
                A.RandomShadow(p=0.3),
            ]
        return A.Compose(base)

    def build(self, requested: int, options: dict[str, Any] | None = None) -> tuple[Path, int]:
        if not self.originals:
            raise ValueError("Upload at least one image before running Auto Mode.")
        requested = max(1, min(int(requested), 500))
        options = options or {}
        width = int(options.get("width") or 512)
        height = int(options.get("height") or 512)
        width = max(64, min(width, 2048))
        height = max(64, min(height, 2048))
        preset = (options.get("preset") or "balanced").lower()
        pipeline = self._compose_pipeline(preset, width, height)
        dataset_dir = self.session_path / f"auto_dataset_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        dataset_dir.mkdir(exist_ok=True)
        generated = 0
        while generated < requested:
            original_path = random.choice(self.originals)
            image = cv2.imread(str(original_path))
            if image is None:
                continue
            augmented = pipeline(image=image)["image"]
            filename = dataset_dir / f"auto_{generated:04d}.png"
            cv2.imwrite(str(filename), augmented)
            generated += 1
        zip_path = dataset_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
            for file in sorted(dataset_dir.glob("*.png")):
                bundle.write(file, arcname=file.name)
        return zip_path, generated
