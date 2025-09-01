# ComfyUI-SeedEdit

Custom ComfyUI nodes for ByteDance SeedEdit image-to-image editing model.

## Features

- **SeedEdit Image to Image**: Edit single images with text prompts
- **SeedEdit Batch Image to Image**: Edit multiple images with the same prompt
- Support for various image sizes and aspect ratios
- Configurable guidance scale and seed control
- Optional watermark removal

## Installation

1. Clone this repository to your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-SeedEdit.git
```

2. Install dependencies:
```bash
cd ComfyUI-SeedEdit
pip install -r requirements.txt
```

3. Configure your API key:
   - Copy `.env.example` to `.env`
   - Add your BytePlus API key to the `.env` file

4. Restart ComfyUI

## Usage

### SeedEdit Image to Image
- **Input**: Image tensor and text prompt
- **Output**: Edited image tensor
- **Parameters**: 
  - `prompt`: Text description of desired edits
  - `guidance_scale`: Control strength of text guidance (1.0-10.0)
  - `seed`: Random seed for reproducible results
  - `watermark`: Whether to add AI-generated watermark

### SeedEdit Batch Image to Image
- **Input**: Batch of image tensors and text prompt
- **Output**: Batch of edited image tensors
- **Features**: Processes multiple images with the same prompt

## API Documentation

For detailed API documentation, visit: https://docs.byteplus.com/en/docs/ModelArk/1666946

## License

MIT License