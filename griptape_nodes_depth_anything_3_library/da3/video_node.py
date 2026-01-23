import logging
import tempfile
import uuid
from pathlib import Path

import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip
from PIL import Image

from griptape.artifacts import VideoUrlArtifact

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from da3.parameters import DepthAnything3Parameters

logger = logging.getLogger("depth_anything_3_library")


def load_video_frames(video_path: str) -> tuple[list[Image.Image], float]:
    """Load video frames as PIL Images and return fps."""
    clip = VideoFileClip(video_path)
    fps = clip.fps
    frames = []
    for frame in clip.iter_frames():
        frames.append(Image.fromarray(frame))
    clip.close()
    return frames, fps


def export_frames_to_video(frames: list[Image.Image], output_path: str, fps: float = 16) -> None:
    """Export PIL Image frames to video file."""
    # Convert PIL Images to numpy arrays
    frame_arrays = [np.array(frame) for frame in frames]
    clip = ImageSequenceClip(frame_arrays, fps=fps)
    clip.write_videofile(str(output_path), codec="libx264", audio=False, logger=None)
    clip.close()


class DepthAnything3Video(ControlNode):
    """Depth estimation for videos using Depth Anything 3."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.params = DepthAnything3Parameters(self)
        self.params.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="input_video",
                input_types=["VideoArtifact", "VideoUrlArtifact"],
                type="VideoArtifact",
                tooltip="Input video for depth estimation.",
            )
        )
        self.add_parameter(
            ParameterBool(
                name="colorize",
                default_value=False,
                tooltip="Output colorized depth video instead of grayscale.",
            )
        )
        self.add_parameter(
            Parameter(
                name="output",
                output_type="VideoUrlArtifact",
                tooltip="Depth video output (grayscale or colorized based on toggle).",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        return self.params.validate_before_node_run()

    def process(self) -> AsyncResult | None:
        yield lambda: self._process()

    def _process(self) -> None:
        input_video_artifact = self.get_parameter_value("input_video")
        colorize = self.get_parameter_value("colorize")

        if input_video_artifact is None:
            msg = "Input video is required"
            raise ValueError(msg)

        logger.info("Loading video frames...")
        input_frames, fps = load_video_frames(input_video_artifact.value)

        if not input_frames:
            msg = "Could not load frames from input video"
            raise ValueError(msg)

        # Create preview placeholder
        first_frame = input_frames[0].convert("RGB")
        preview_placeholder = self._create_placeholder_video([first_frame], fps)
        self.publish_update_to_parameter("output", preview_placeholder)

        logger.info(f"Processing {len(input_frames)} frames...")

        # Process each frame
        output_frames = []

        for i, frame in enumerate(input_frames):
            frame_rgb = frame.convert("RGB")
            depth, original_size = self.params.process_depth_estimation(frame_rgb)

            if colorize:
                output_frame = self.params.depth_to_colorized_pil(depth, original_size)
            else:
                output_frame = self.params.depth_to_grayscale_pil(depth, original_size)
                output_frame = output_frame.convert("RGB")  # Convert grayscale to RGB for video export

            output_frames.append(output_frame)

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Processed frame {i + 1}/{len(input_frames)}")

        logger.info("Exporting depth video...")

        # Export depth video
        output_artifact = self._save_video(output_frames, "depth", fps)

        # Set output
        self.set_parameter_value("output", output_artifact)
        self.parameter_output_values["output"] = output_artifact

        logger.info("Video depth estimation complete.")

    def _create_placeholder_video(self, frames: list[Image.Image], fps: float) -> VideoUrlArtifact:
        """Create a placeholder video for preview purposes."""
        placeholder_frames = [self.params.create_preview_placeholder(frame.size) for frame in frames]
        return self._save_video(placeholder_frames, "placeholder", fps)

    def _save_video(self, frames: list[Image.Image], prefix: str, fps: float) -> VideoUrlArtifact:
        """Export frames to video and return artifact."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file_obj:
            temp_file = Path(temp_file_obj.name)

        try:
            export_frames_to_video(frames, str(temp_file), fps=fps)
            filename = f"{prefix}_{uuid.uuid4().hex[:8]}.mp4"
            url = GriptapeNodes.StaticFilesManager().save_static_file(temp_file.read_bytes(), filename)
            return VideoUrlArtifact(url)
        finally:
            if temp_file.exists():
                temp_file.unlink()
