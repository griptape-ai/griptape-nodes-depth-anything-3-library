import io
import logging
import tempfile
import uuid
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

logger = logging.getLogger("depth_anything_3_library")

AVAILABLE_MODELS = [
    "depth-anything/DA3-LARGE-1.1",
    "depth-anything/DA3-GIANT-1.1",
    "depth-anything/DA3-LARGE",
    "depth-anything/DA3-GIANT",
    "depth-anything/DA3-BASE",
    "depth-anything/DA3-SMALL",
]


class DepthAnythingNode(SuccessFailureNode):
    """Node for depth estimation using Depth Anything 3."""

    _model = None
    _current_model_name = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._model_repo_parameter = HuggingFaceRepoParameter(
            self,
            repo_ids=AVAILABLE_MODELS,
            parameter_name="model",
        )
        self._model_repo_parameter.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="image",
                allowed_modes={ParameterMode.INPUT},
                type="ImageArtifact",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                default_value=None,
                tooltip="Input image for depth estimation.",
            )
        )

        self.add_parameter(
            Parameter(
                name="depth",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="ImageArtifact",
                default_value=None,
                tooltip="Grayscale depth map (closer = brighter).",
            )
        )

        self.add_parameter(
            Parameter(
                name="depth_colorized",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="ImageArtifact",
                default_value=None,
                tooltip="Colorized depth map for visualization.",
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the depth estimation result",
            result_details_placeholder="Depth estimation result details will appear here.",
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate that the HuggingFace model is available."""
        return self._model_repo_parameter.validate_before_node_run()

    def _get_device(self) -> str:
        """Get the appropriate device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self, model_id: str) -> None:
        """Load the Depth Anything 3 model from HuggingFace Hub."""
        import safetensors  # noqa: F401 - needed for huggingface_hub bug workaround

        from depth_anything_3.api import DepthAnything3

        if DepthAnythingNode._model is not None and DepthAnythingNode._current_model_name == model_id:
            logger.info(f"Using cached model: {model_id}")
            return

        logger.info(f"Loading Depth Anything 3 model: {model_id}")
        device = torch.device(self._get_device())

        DepthAnythingNode._model = DepthAnything3.from_pretrained(model_id)
        DepthAnythingNode._model = DepthAnythingNode._model.to(device=device)
        DepthAnythingNode._current_model_name = model_id

        logger.info(f"Model loaded on device: {device}")

    def _image_artifact_to_pil(self, artifact: ImageArtifact | ImageUrlArtifact) -> Image.Image:
        """Convert an ImageArtifact or ImageUrlArtifact to a PIL Image."""
        if isinstance(artifact, ImageUrlArtifact):
            response = requests.get(artifact.value)
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        return Image.open(io.BytesIO(artifact.value)).convert("RGB")

    def _pil_to_artifact(self, pil_image: Image.Image, prefix: str = "depth") -> ImageUrlArtifact:
        """Convert a PIL Image to an ImageUrlArtifact using StaticFilesManager."""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        image_bytes = buffer.read()

        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        url = GriptapeNodes.StaticFilesManager().save_static_file(image_bytes, filename)

        return ImageUrlArtifact(value=url)

    def _depth_to_grayscale_pil(self, depth: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
        """Convert depth array to grayscale PIL Image."""
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_inverted = 1.0 - depth_normalized
        depth_uint8 = (depth_inverted * 255).astype(np.uint8)

        depth_image = Image.fromarray(depth_uint8, mode="L")
        return depth_image.resize(original_size, Image.Resampling.LANCZOS)

    def _depth_to_colorized_pil(self, depth: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
        """Convert depth array to colorized PIL Image using a colormap."""
        import cv2

        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        depth_image = Image.fromarray(depth_colored_rgb)
        return depth_image.resize(original_size, Image.Resampling.LANCZOS)

    async def aprocess(self) -> None:
        await self._process()

    async def _process(self) -> None:
        self._clear_execution_status()

        model_repo_id = self.get_parameter_value("model")
        image_artifact = self.get_parameter_value("image")

        if image_artifact is None:
            self._set_status_results(
                was_successful=False,
                result_details="No image input provided.",
            )
            return

        try:
            self.status_component.append_to_result_details(f"Loading model: {model_repo_id}")
            self._load_model(model_repo_id)

            model = DepthAnythingNode._model

            pil_image = self._image_artifact_to_pil(image_artifact)
            original_size = pil_image.size

            self.status_component.append_to_result_details(f"Processing image ({original_size[0]}x{original_size[1]})...")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_image_path = Path(temp_dir) / "input.png"
                pil_image.save(temp_image_path)

                with torch.inference_mode():
                    prediction = model.inference(
                        image=[str(temp_image_path)],
                        process_res=504,
                        process_res_method="upper_bound_resize",
                    )

            depth = prediction.depth[0]

            depth_pil = self._depth_to_grayscale_pil(depth, original_size)
            depth_colorized_pil = self._depth_to_colorized_pil(depth, original_size)

            depth_artifact = self._pil_to_artifact(depth_pil, "depth")
            depth_colorized_artifact = self._pil_to_artifact(depth_colorized_pil, "depth_colorized")

            self.set_parameter_value("depth", depth_artifact)
            self.set_parameter_value("depth_colorized", depth_colorized_artifact)
            self.parameter_output_values["depth"] = depth_artifact
            self.parameter_output_values["depth_colorized"] = depth_colorized_artifact

            self._set_status_results(
                was_successful=True,
                result_details=f"Depth estimation complete. Output size: {original_size[0]}x{original_size[1]}",
            )

        except Exception as e:
            msg = f"Error during depth estimation: {e!s}"
            logger.exception(msg)
            self._set_status_results(was_successful=False, result_details=msg)
            return
