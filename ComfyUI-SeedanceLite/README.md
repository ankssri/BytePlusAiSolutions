# ComfyUI-Seedance-lite

Custom ComfyUI nodes for ByteDance Seedance-lite video generation models.

## Features

- **Seedance-lite Text to Video**: Generate videos from text prompts
- **Seedance-lite Image to Video**: Generate videos from first frame + optional last frame
- **Seedance-lite Reference Images to Video**: Generate videos from 1-4 reference images
- **Seedance-lite Task Query**: Query the status of video generation tasks
- Support for various resolutions (480p, 720p) and aspect ratios
- Configurable video parameters via text commands

## Installation

1. Clone this repository to your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-Seedance-lite.git
```

2. Install dependencies:
```bash
cd ComfyUI-Seedance-lite
pip install -r requirements.txt
```

3. Configure your API key:
   - Copy `.env.example` to `.env`
   - Add your BytePlus API key to the `.env` file

4. Restart ComfyUI

## Usage

### Text to Video
- **Input**: Text prompt
- **Output**: Task ID for video generation
- **Parameters**: Resolution, aspect ratio, callback URL

### Image to Video
- **Input**: First frame image, optional last frame image, optional text prompt
- **Output**: Task ID for video generation
- **Features**: Supports first-frame + last-frame video generation

### Reference Images to Video
- **Input**: 1-4 reference images, optional text prompt
- **Output**: Task ID for video generation
- **Features**: Uses reference images to guide video generation

### Task Query
- **Input**: Task ID from video generation
- **Output**: Status, video URL (when ready), full result JSON
- **Features**: Check generation progress and retrieve video URLs

## Workflow

1. Use one of the generation nodes to create a video generation task
2. Use the Task Query node to check the status
3. When status is "succeeded", the video URL will be available
4. Download the video from the provided URL

## API Documentation

For detailed API documentation, visit: https://docs.byteplus.com/en/docs/ModelArk/1520757

## License

MIT License