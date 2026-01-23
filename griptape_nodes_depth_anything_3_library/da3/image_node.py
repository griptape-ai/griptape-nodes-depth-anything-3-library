import logging

from griptape.artifacts import ImageUrlArtifact

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from da3.parameters import DepthAnything3Parameters

logger = logging.getLogger("depth_anything_3_library")


class DepthAnything3Image(ControlNode):
    """Depth estimation for images using Depth Anything 3."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.params = DepthAnything3Parameters(self)
        self.params.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="input_image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Input image for depth estimation.",
            )
        )
        self.add_parameter(
            ParameterBool(
                name="colorize",
                default_value=False,
                tooltip="Output colorized depth map instead of grayscale.",
            )
        )
        self.add_parameter(
            Parameter(
                name="output",
                output_type="ImageArtifact",
                tooltip="Depth map output (grayscale or colorized based on toggle).",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        return self.params.validate_before_node_run()

    def process(self) -> AsyncResult | None:
        yield lambda: self._process()

    def _process(self) -> None:
        input_image_artifact = self.get_parameter_value("input_image")

        if input_image_artifact is None:
            msg = "Input image is required"
            raise ValueError(msg)

        # Convert to PIL
        input_image_pil = self.params.image_artifact_to_pil(input_image_artifact)
        colorize = self.get_parameter_value("colorize")

        # Set preview placeholder
        preview_placeholder = self.params.create_preview_placeholder(input_image_pil.size)
        self.publish_update_to_parameter("output", self.params.pil_to_image_artifact(preview_placeholder, "depth"))

        logger.info(f"Processing image ({input_image_pil.size[0]}x{input_image_pil.size[1]})...")

        # Process depth estimation
        depth, original_size = self.params.process_depth_estimation(input_image_pil)

        # Convert to output image based on colorize toggle
        if colorize:
            output_pil = self.params.depth_to_colorized_pil(depth, original_size)
        else:
            output_pil = self.params.depth_to_grayscale_pil(depth, original_size)

        # Create artifact
        output_artifact = self.params.pil_to_image_artifact(output_pil, "depth")

        # Set output
        self.set_parameter_value("output", output_artifact)
        self.parameter_output_values["output"] = output_artifact

        logger.info(f"Depth estimation complete. Output size: {original_size[0]}x{original_size[1]}")
