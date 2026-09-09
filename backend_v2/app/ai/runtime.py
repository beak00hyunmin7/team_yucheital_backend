from __future__ import annotations

import contextlib
import hashlib
import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

RAIN_MIN_MM = 20.0
RAIN_MAX_MM = 80.0
TIME_MIN_S = 0.0
TIME_MAX_S = 120.0
DEPTH_VMAX_M = 0.6


class ModelUnavailable(RuntimeError):
    pass


class ModelContractError(ModelUnavailable):
    pass


@dataclass(frozen=True)
class RuntimeStatus:
    available: bool
    model_version: str
    checkpoint_hash: str | None
    checkpoint_name: str
    device: str | None
    condition_dimension: int | None
    error: str | None


@dataclass(frozen=True)
class ModelPrediction:
    water_map_png: bytes
    depth_m: np.ndarray
    inference_time_ms: int
    checkpoint_hash: str
    device: str


class DrainageModelRuntime:
    """Single-load, bounded-concurrency runtime for the supplied FiLM model.

    It never falls back to D8 or any heuristic model. An unavailable or
    incompatible checkpoint is surfaced to readiness and prediction callers.
    """

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        model_version: str,
        image_size: int,
        concurrency: int,
        use_fp16: bool,
        warmup: bool,
        wait_timeout_s: float = 30.0,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.model_version = model_version
        self.image_size = image_size
        self.use_fp16 = use_fp16
        self.warmup = warmup
        self.wait_timeout_s = wait_timeout_s
        self._model = None
        self._torch = None
        self._device: str | None = None
        self._checkpoint_hash: str | None = None
        self._cond_dim: int | None = None
        self._error: str | None = None
        self._failure_recorded = False
        self._failed_signature: tuple[int, int] | None = None
        self._load_lock = threading.Lock()
        self._inference_slots = threading.BoundedSemaphore(max(1, concurrency))
        self._lut = (
            np.asarray(
                [matplotlib.colormaps["Blues"](index / 255.0)[:3] for index in range(256)]
            )
            * 255.0
        ).astype(np.float32)
        self._lut_tree = None

    def try_load(self) -> RuntimeStatus:
        try:
            self.load()
        except ModelUnavailable:
            pass
        return self.status()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            signature = _file_signature(self.checkpoint_path)
            if (
                self._failure_recorded
                and signature == self._failed_signature
                and self._error is not None
            ):
                raise ModelUnavailable(self._error)
            try:
                if not self.checkpoint_path.is_file():
                    raise ModelUnavailable(
                        f"checkpoint not found: {self.checkpoint_path.name}"
                    )
                try:
                    import torch
                except ImportError as exc:
                    raise ModelUnavailable("PyTorch is not installed") from exc

                from app.ai.unet_generator import UnetGenerator

                device = "cuda" if torch.cuda.is_available() else "cpu"
                state = torch.load(
                    self.checkpoint_path,
                    map_location=device,
                    weights_only=True,
                )
                generator_state = state.get("netG")
                if not isinstance(generator_state, dict):
                    raise ModelContractError("checkpoint does not contain netG weights")

                outer_key = self._find_outer_weight(generator_state)
                input_channels = int(generator_state[outer_key].shape[1])
                ngf = int(generator_state[outer_key].shape[0])
                film_keys = [
                    key for key in generator_state if key.endswith("film.net.0.weight")
                ]
                cond_dim = int(generator_state[film_keys[0]].shape[1]) if film_keys else 0
                if input_channels != 3 or cond_dim != 2:
                    raise ModelContractError(
                        "v2 requires the v5 contract: RGB input and FiLM cond_dim=2; "
                        f"checkpoint reports in_channels={input_channels}, cond_dim={cond_dim}"
                    )

                model = UnetGenerator(
                    in_channels=input_channels,
                    out_channels=3,
                    image_size=self.image_size,
                    ngf=ngf,
                    use_film=True,
                    cond_dim=cond_dim,
                ).to(device)
                model.load_state_dict(generator_state, strict=True)
                model.eval()
                if device == "cuda":
                    torch.backends.cudnn.benchmark = True

                self._torch = torch
                self._model = model
                self._device = device
                self._cond_dim = cond_dim
                self._checkpoint_hash = _sha256_file(self.checkpoint_path)
                self._error = None
                self._failure_recorded = False
                self._failed_signature = None
                if self.warmup:
                    self._warmup()
            except ModelUnavailable as exc:
                self._reset_loaded_state()
                self._error = str(exc)
                self._failure_recorded = True
                self._failed_signature = signature
                raise
            except Exception as exc:
                self._reset_loaded_state()
                self._error = f"model load failed: {type(exc).__name__}: {exc}"
                self._failure_recorded = True
                self._failed_signature = signature
                raise ModelUnavailable(self._error) from exc

    def predict(self, model_input_png: bytes, *, rain_mm: float, time_s: float) -> ModelPrediction:
        self.load()
        if self._model is None or self._torch is None or self._device is None:
            raise ModelUnavailable(self._error or "model is not loaded")
        started_at = time.perf_counter()
        acquired = self._inference_slots.acquire(timeout=self.wait_timeout_s)
        if not acquired:
            raise ModelUnavailable("AI inference queue is full; retry later")
        try:
            input_tensor = self._to_tensor(model_input_png)
            condition = self._condition_tensor(rain_mm, time_s)
            torch = self._torch
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self._device == "cuda" and self.use_fp16
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), autocast_context:
                prediction = self._model(input_tensor, condition)
            rgb = _tensor_to_rgb(prediction[0])
        finally:
            self._inference_slots.release()
        elapsed_ms = max(1, round((time.perf_counter() - started_at) * 1000))
        water_map_png = _rgb_to_png(rgb)
        depth = self._rgb_to_depth(rgb)
        return ModelPrediction(
            water_map_png=water_map_png,
            depth_m=depth.astype(np.float32),
            inference_time_ms=elapsed_ms,
            checkpoint_hash=self._checkpoint_hash or "",
            device=self._device,
        )

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(
            available=self._model is not None,
            model_version=self.model_version,
            checkpoint_hash=self._checkpoint_hash,
            checkpoint_name=self.checkpoint_path.name,
            device=self._device,
            condition_dimension=self._cond_dim,
            error=self._error,
        )

    def _to_tensor(self, png: bytes):
        image = Image.open(io.BytesIO(png)).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        values = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        values = np.transpose(values, (2, 0, 1)).copy()
        return self._torch.from_numpy(values).unsqueeze(0).to(self._device)

    def _condition_tensor(self, rain_mm: float, time_s: float):
        rain = (rain_mm - RAIN_MIN_MM) / (RAIN_MAX_MM - RAIN_MIN_MM)
        elapsed = (time_s - TIME_MIN_S) / (TIME_MAX_S - TIME_MIN_S)
        values = [[float(np.clip(rain, 0.0, 1.0)), float(np.clip(elapsed, 0.0, 1.0))]]
        return self._torch.tensor(values, dtype=self._torch.float32, device=self._device)

    def _warmup(self) -> None:
        torch = self._torch
        if torch is None or self._model is None or self._device is None:
            return
        sample = torch.zeros(
            (1, 3, self.image_size, self.image_size),
            dtype=torch.float32,
            device=self._device,
        )
        condition = torch.zeros((1, 2), dtype=torch.float32, device=self._device)
        with torch.inference_mode():
            self._model(sample, condition)
        if self._device == "cuda":
            torch.cuda.synchronize()

    def _rgb_to_depth(self, rgb: np.ndarray) -> np.ndarray:
        flat = rgb.reshape(-1, 3).astype(np.float32)
        try:
            if self._lut_tree is None:
                from scipy.spatial import cKDTree

                self._lut_tree = cKDTree(self._lut)
            _, indices = self._lut_tree.query(flat, workers=-1)
        except ImportError:
            indices = _nearest_lut_chunked(flat, self._lut)
        normalized = indices.reshape(rgb.shape[:2]).astype(np.float32) / 255.0
        return (np.clip(normalized, 0.0, 1.0) ** 2) * DEPTH_VMAX_M

    @staticmethod
    def _find_outer_weight(generator_state: dict[str, object]) -> str:
        for candidate in ("model.down.0.weight", "model.model.0.weight"):
            if candidate in generator_state:
                return candidate
        raise ModelContractError("unknown U-Net state-dict layout")

    def _reset_loaded_state(self) -> None:
        self._model = None
        self._torch = None
        self._device = None
        self._checkpoint_hash = None
        self._cond_dim = None


def _nearest_lut_chunked(flat: np.ndarray, lut: np.ndarray) -> np.ndarray:
    output = np.empty(flat.shape[0], dtype=np.int16)
    chunk_size = 8192
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start : start + chunk_size]
        distances = ((chunk[:, None, :] - lut[None, :, :]) ** 2).sum(axis=2)
        output[start : start + chunk.shape[0]] = distances.argmin(axis=1)
    return output


def _tensor_to_rgb(tensor) -> np.ndarray:
    values = tensor.detach().float().cpu().clamp(-1.0, 1.0).numpy()
    values = ((values + 1.0) * 127.5).round().astype(np.uint8)
    return np.transpose(values, (1, 2, 0))


def _rgb_to_png(rgb: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG", compress_level=3)
    return output.getvalue()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns
