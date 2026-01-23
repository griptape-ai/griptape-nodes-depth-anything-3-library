import io
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from griptape.artifacts import ImageArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.traits.options import Options

logger = logging.getLogger("depth_anything_3_library")

MODEL_CHOICES = [
    "da3-large",
    "da3-giant",
    "da3-base",
    "da3-small",
]


class DepthAnythingNode(SuccessFailureNode):
    """Node for depth estimation using Depth Anything 3."""

    _model = None
    _current_model_name = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="model",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="da3-large",
                tooltip="Model variant to use for depth estimation.",
                traits={Options(choices=MODEL_CHOICES)},
            )
        )

        self.add_parameter(
            Parameter(
                name="image",
                allowed_modes={ParameterMode.INPUT},
                type="ImageArtifact",
                input_types=["ImageArtifact"],
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

    def _get_device(self) -> str:
        """Get the appropriate device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self, model_name: str) -> None:
        """Load the Depth Anything 3 model."""
        from depth_anything_3.api import DepthAnything3

        if DepthAnythingNode._model is not None and DepthAnythingNode._current_model_name == model_name:
            logger.info(f"Using cached model: {model_name}")
            return

        logger.info(f"Loading Depth Anything 3 model: {model_name}")
        device = self._get_device()

        DepthAnythingNode._model = DepthAnything3(model_name=model_name)
        DepthAnythingNode._model = DepthAnythingNode._model.eval().to(device)
        DepthAnythingNode._current_model_name = model_name

        logger.info(f"Model loaded on device: {device}")

    def _image_artifact_to_pil(self, artifact: ImageArtifact) -> Image.Image:
        """Convert an ImageArtifact to a PIL Image."""
        buffer = io.BytesIO(artifact.value)
        return Image.open(buffer).convert("RGB")

    def _depth_to_grayscale_artifact(self, depth: np.ndarray, original_size: tuple[int, int]) -> ImageArtifact:
        """Convert depth array to grayscale ImageArtifact."""
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_inverted = 1.0 - depth_normalized
        depth_uint8 = (depth_inverted * 255).astype(np.uint8)

        depth_image = Image.fromarray(depth_uint8, mode="L")
        depth_image = depth_image.resize(original_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        depth_image.save(buffer, format="PNG")
        buffer.seek(0)

        return ImageArtifact(value=buffer.read(), format="png", width=original_size[0], height=original_size[1])

    def _depth_to_colorized_artifact(self, depth: np.ndarray, original_size: tuple[int, int]) -> ImageArtifact:
        """Convert depth array to colorized ImageArtifact using a colormap."""
        import cv2

        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        depth_image = Image.fromarray(depth_colored_rgb)
        depth_image = depth_image.resize(original_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        depth_image.save(buffer, format="PNG")
        buffer.seek(0)

        return ImageArtifact(value=buffer.read(), format="png", width=original_size[0], height=original_size[1])

    async def aprocess(self) -> None:
        await self._process()

    async def _process(self) -> None:
        self._clear_execution_status()

        model_name = self.get_parameter_value("model")
        image_artifact = self.get_parameter_value("image")

        if image_artifact is None:
            self._set_status_results(
                was_successful=False,
                result_details="No image input provided.",
            )
            return

        try:
            self.status_component.append_to_result_details(f"Loading model: {model_name}")
            self._load_model(model_name)

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

            depth_artifact = self._depth_to_grayscale_artifact(depth, original_size)
            depth_colorized_artifact = self._depth_to_colorized_artifact(depth, original_size)

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
