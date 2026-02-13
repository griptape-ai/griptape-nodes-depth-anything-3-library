# Griptape Nodes: Depth Anything 3 Library

A [Griptape Nodes](https://www.griptapenodes.com/) library for monocular depth estimation using [Depth Anything 3](https://github.com/LiheYoung/Depth-Anything) models.

## Overview

This library provides nodes for estimating depth from images and videos using state-of-the-art Depth Anything 3 models. The depth maps can be output as grayscale images (closer = brighter) or colorized using the INFERNO colormap.

## Requirements

- **GPU**: CUDA (NVIDIA) or MPS (Apple Silicon) is required for inference
- **Griptape Nodes Engine**: Version 0.66.2 or later

## Nodes

### Depth Anything 3 Image

Estimates depth from a single image.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | Dropdown | Select the Depth Anything 3 model to use (see [Available Models](#available-models)) |
| `input_image` | ImageArtifact | Input image for depth estimation |
| `colorize` | Boolean | When enabled, outputs a colorized depth map using the INFERNO colormap. When disabled, outputs a grayscale depth map where closer objects are brighter. Default: `False` |
| `output` | ImageArtifact | The resulting depth map |

### Depth Anything 3 Video

Estimates depth from a video, processing each frame independently.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | Dropdown | Select the Depth Anything 3 model to use (see [Available Models](#available-models)) |
| `input_video` | VideoArtifact | Input video for depth estimation |
| `colorize` | Boolean | When enabled, outputs a colorized depth video using the INFERNO colormap. When disabled, outputs a grayscale depth video where closer objects are brighter. Default: `False` |
| `output` | VideoUrlArtifact | The resulting depth video |

## Available Models

The following Depth Anything 3 models are available from HuggingFace:

| Model | Description |
|-------|-------------|
| `depth-anything/DA3-SMALL` | Smallest and fastest model |
| `depth-anything/DA3-BASE` | Base model, good balance of speed and quality |
| `depth-anything/DA3-LARGE` | Large model with improved accuracy |
| `depth-anything/DA3-LARGE-1.1` | Improved large model (v1.1) |
| `depth-anything/DA3-GIANT` | Highest quality, slowest inference |
| `depth-anything/DA3-GIANT-1.1` | Improved giant model (v1.1) |

Models are cached after first use, so subsequent runs with the same model will be faster.

## Installation

### Prerequisites

- [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes) installed and running
- A CUDA-capable NVIDIA GPU or Apple Silicon Mac

### Install the Library

1. **Clone the repository** to your Griptape Nodes workspace directory:

   ```bash
   # Navigate to your Griptape Nodes workspace directory
   cd `gtn config show workspace_directory`

   # Clone the repository with submodules
   git clone --recurse-submodules https://github.com/griptape-ai/griptape-nodes-depth-anything-3-library.git
   ```

2. **Add the library** in the Griptape Nodes Editor:

   - Open the Settings menu and navigate to the *Libraries* settings
   - Click on *+ Add Library* at the bottom of the settings panel
   - Enter the path to the library JSON file:
     ```
     <workspace_directory>/griptape-nodes-depth-anything-3-library/griptape_nodes_depth_anything_3_library/griptape-nodes-library.json
     ```
   - You can check your workspace directory with `gtn config show workspace_directory`
   - Close the Settings Panel
   - Click on *Refresh Libraries*

3. **Verify installation** by checking that the "Depth Anything 3 Image" and "Depth Anything 3 Video" nodes appear in the node palette under the "Depth/Estimation" category.

## Usage

### Image Depth Estimation

1. Add a **Depth Anything 3 Image** node to your workflow
2. Connect an image source to the `input_image` parameter
3. Select a model from the `model` dropdown
4. Optionally enable `colorize` for a color depth map
5. Connect the `output` to your next node or a display

### Video Depth Estimation

1. Add a **Depth Anything 3 Video** node to your workflow
2. Connect a video source to the `input_video` parameter
3. Select a model from the `model` dropdown
4. Optionally enable `colorize` for a color depth video
5. Connect the `output` to your next node or a display

Note: Video processing can take significant time depending on the video length and model size.

## Output Format

- **Grayscale mode** (`colorize: False`): Outputs a single-channel image where brightness indicates proximity. Closer objects appear brighter (white), distant objects appear darker (black).

- **Colorized mode** (`colorize: True`): Outputs an RGB image using the INFERNO colormap, where colors range from dark purple/black (far) through red/orange (mid) to bright yellow (near).

## Troubleshooting

### Library Not Loading

- Ensure the git submodule is initialized. If you cloned without `--recurse-submodules`, run:
  ```bash
  git submodule update --init --recursive
  ```

### CUDA/MPS Not Available

- Verify your GPU drivers are up to date
- For NVIDIA GPUs, ensure CUDA is properly installed
- For Apple Silicon, ensure you're running on macOS 12.3 or later

### Out of Memory Errors

- Try using a smaller model (e.g., `DA3-SMALL` or `DA3-BASE`)
- For videos, consider processing shorter clips
- Close other GPU-intensive applications

## Additional Resources

- [Depth Anything 3 Paper](https://arxiv.org/abs/2406.09414)
- [Depth Anything GitHub](https://github.com/LiheYoung/Depth-Anything)
- [Griptape Nodes Documentation](https://docs.griptapenodes.com/)
- [Griptape Discord](https://discord.gg/griptape)

## License

This library is provided under the Apache License 2.0.
