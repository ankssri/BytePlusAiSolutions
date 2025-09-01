# ComfyUI-Seedream

A custom ComfyUI node package for ByteDance Seedream text-to-image generation API integration.

## Overview

ComfyUI-Seedream provides seamless integration with ByteDance's Seedream text-to-image generation API, offering three powerful nodes for different image generation workflows:

- **SeedreamTextToImage**: Single image generation from text prompts
- **SeedreamBatchTextToImage**: Batch processing multiple prompts simultaneously
- **SeedreamVariationsTextToImage**: Generate multiple variations from a single prompt

## Features

- 🎨 High-quality text-to-image generation using ByteDance Seedream models
- 📦 Batch processing capabilities for efficient workflow
- 🎲 Variation generation with seed control
- 🔧 Configurable image sizes and generation parameters
- 🔑 Environment-based API key management
- 💧 Optional watermark control
- 🎯 Guidance scale adjustment for prompt adherence

## Installation

1. Clone or download this repository to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone <repository-url> ComfyUI-Seedream

2. cd ComfyUI-Seedream
pip install -r requirements.txt

3. Create .env file
SEEDREAM_API_KEY=your_api_key_here
SEEDREAM_MODEL=your_preferred_model_id



## Usage Examples
### Basic Text-to-Image Generation
1. 1.
   Add a "Seedream Text to Image" node to your workflow
2. 2.
   Connect a text input with your prompt
3. 3.
   Configure the desired image size and parameters
4. 4.
   Connect the output to a "Save Image" node
### Batch Processing
1. 1.
   Use "Seedream Batch Text to Image" for multiple prompts
2. 2.
   Input prompts separated by newlines
3. 3.
   Set appropriate batch size based on your needs
4. 4.
   Process all generated images through your workflow
### Creating Variations
1. 1.
   Use "Seedream Variations Text to Image" for exploring different interpretations
2. 2.
   Set the number of variations you want
3. 3.
   Use a fixed base seed for consistent variation patterns
## Supported Image Sizes
- Square : 1024x1024
- Portrait : 864x1152, 720x1280, 832x1248
- Landscape : 1152x864, 1280x720, 1248x832, 1512x648
## Requirements
- Python 3.8+
- ComfyUI
- PyTorch 1.12.0+
