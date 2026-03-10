import io
import logging
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.files.file import File, FileLoadError

logger = logging.getLogger("depth_anything_3_library")

AVAILABLE_MODELS = [
    "depth-anything/DA3-SMALL",
    "depth-anything/DA3-BASE",
    "depth-anything/DA3-LARGE",
    "depth-anything/DA3-LARGE-1.1",
    "depth-anything/DA3-GIANT",
    "depth-anything/DA3-GIANT-1.1",
]


class DepthAnything3Parameters:
    """Shared parameters and utilities for Depth Anything 3 nodes."""

    _model = None
    _current_model_name = None

    def __init__(self, node: BaseNode):
        self._node = node
        self._huggingface_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=AVAILABLE_MODELS,
            parameter_name="model",
        )

    def add_input_parameters(self) -> None:
        self._huggingface_repo_parameter.add_input_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        return self._huggingface_repo_parameter.validate_before_node_run()

    def get_model_id(self) -> str:
        return self._node.get_parameter_value("model")

    @staticmethod
    def get_device() -> str:
        """Get the appropriate device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self):
        """Load the Depth Anything 3 model from HuggingFace Hub."""
        from depth_anything_3.api import DepthAnything3

        model_id = self.get_model_id()

        if DepthAnything3Parameters._model is not None and DepthAnything3Parameters._current_model_name == model_id:
            logger.info(f"Using cached model: {model_id}")
            return DepthAnything3Parameters._model

        logger.info(f"Loading Depth Anything 3 model: {model_id}")
        device = torch.device(self.get_device())

        DepthAnything3Parameters._model = DepthAnything3.from_pretrained(model_id)
        DepthAnything3Parameters._model = DepthAnything3Parameters._model.to(device=device)
        DepthAnything3Parameters._current_model_name = model_id

        logger.info(f"Model loaded on device: {device}")
        return DepthAnything3Parameters._model

    def process_depth_estimation(self, pil_image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
        """Process a single image for depth estimation.

        Returns:
            Tuple of (depth array, original size)
        """
        model = self.load_model()
        original_size = pil_image.size

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_image_path = Path(temp_dir) / "input.png"
            pil_image.save(temp_image_path)

            with torch.inference_mode():
                prediction = model.inference(
                    image=[str(temp_image_path)],
                    process_res=504,
                    process_res_method="upper_bound_resize",
                )

        return prediction.depth[0], original_size

    def depth_to_grayscale_pil(self, depth: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
        """Convert depth array to grayscale PIL Image (closer = brighter)."""
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_inverted = 1.0 - depth_normalized
        depth_uint8 = (depth_inverted * 255).astype(np.uint8)

        depth_image = Image.fromarray(depth_uint8, mode="L")
        return depth_image.resize(original_size, Image.Resampling.LANCZOS)

    def depth_to_colorized_pil(self, depth: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
        """Convert depth array to colorized PIL Image using INFERNO colormap."""
        import cv2

        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        depth_image = Image.fromarray(depth_colored_rgb)
        return depth_image.resize(original_size, Image.Resampling.LANCZOS)

    @staticmethod
    def image_artifact_to_pil(artifact: ImageArtifact | ImageUrlArtifact) -> Image.Image:
        """Convert an ImageArtifact or ImageUrlArtifact to a PIL Image."""
        if isinstance(artifact, ImageUrlArtifact):
            image_bytes = File(artifact.value).read_bytes()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return Image.open(io.BytesIO(artifact.value)).convert("RGB")

    @staticmethod
    def pil_to_image_artifact(pil_image: Image.Image, prefix: str = "depth") -> ImageUrlArtifact:
        """Convert a PIL Image to an ImageUrlArtifact using StaticFilesManager."""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        image_bytes = buffer.read()

        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        url = GriptapeNodes.StaticFilesManager().save_static_file(image_bytes, filename)

        return ImageUrlArtifact(value=url)

    @staticmethod
    def create_preview_placeholder(size: tuple[int, int]) -> Image.Image:
        """Create a black placeholder image for preview."""
        return Image.new("RGB", size, color="black")
