import torch
import requests
import base64
from PIL import Image
from io import BytesIO
import time
import json
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get default model from environment variable
DEFAULT_MODEL = os.getenv('SEEDREAM_MODEL', 'seedream-3-0-t2i-250415')

class SeedreamTextToImage:
    """
    Custom ComfyUI node for ByteDance Seedream text-to-image generation
    """
    
    CATEGORY = "image/generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape with mountains and a lake"
                }),
                "model": ([DEFAULT_MODEL], {
                    "default": DEFAULT_MODEL
                }),
                "size": (["1024x1024", "864x1152", "1152x864", "1280x720", "720x1280", "832x1248", "1248x832", "1512x648"], {
                    "default": "1024x1024"
                }),
                "response_format": (["url", "b64_json"], {
                    "default": "b64_json"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "watermark": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    
    def generate_image(self, prompt, model, size, response_format, seed=-1, guidance_scale=2.5, watermark=True):
        """
        Generate image using BytePlus Seedream API
        """
        
        # Load API key from environment variable
        api_key = os.getenv('SEEDREAM_API_KEY')
        if not api_key:
            raise ValueError("SEEDREAM_API_KEY not found in environment variables. Please add it to your .env file.")
        
        if not api_key:
            raise ValueError("API key is required")
        
        # API endpoint
        url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Prepare request data
        data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": response_format,
            "guidance_scale": guidance_scale,
            "watermark": watermark
        }
        
        # Add seed if specified
        if seed != -1:
            data["seed"] = seed
        
        try:
            # Make API request
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "data" not in result or not result["data"]:
                raise ValueError("No image data returned from API")
            
            # Process the response based on format
            if response_format == "b64_json":
                # Decode base64 image
                image_data = result["data"][0]["b64_json"]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
            else:
                # Download image from URL
                image_url = result["data"][0]["url"]
                img_response = requests.get(image_url, timeout=30)
                img_response.raise_for_status()
                image = Image.open(BytesIO(img_response.content))
            
            # Convert PIL image to tensor format expected by ComfyUI
            # ComfyUI expects images in format [B, H, W, C] with values in range [0, 1]
            image = image.convert("RGB")
            image_array = torch.from_numpy(np.array(image)).float() / 255.0
            
            # Add batch dimension
            image_tensor = image_array.unsqueeze(0)
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Image generation failed: {str(e)}")

# Additional utility node for batch generation
class SeedreamBatchTextToImage:
    """
    Batch version of Seedream text-to-image generation
    """
    
    CATEGORY = "image/generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape\nA city skyline\nA forest scene"
                }),
                "model": ([DEFAULT_MODEL], {
                    "default": DEFAULT_MODEL
                }),
                "size": (["1024x1024", "864x1152", "1152x864", "1280x720", "720x1280", "832x1248", "1248x832", "1512x648"], {
                    "default": "1024x1024"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "watermark": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_batch_images"
    
    def generate_batch_images(self, prompts, model, size, batch_size, seed=-1, guidance_scale=2.5, watermark=True):
        """
        Generate multiple images from multiple prompts
        """
        
        # Load API key from environment variable
        api_key = os.getenv('SEEDREAM_API_KEY')
        if not api_key:
            raise ValueError("SEEDREAM_API_KEY not found in environment variables. Please add it to your .env file.")
        
        # Split prompts by newlines
        prompt_list = [p.strip() for p in prompts.split('\n') if p.strip()]
        
        if not prompt_list:
            raise ValueError("No valid prompts provided")
        
        # Limit to batch_size prompts
        prompt_list = prompt_list[:batch_size]
        
        # Create single node instance for reuse
        single_node = SeedreamTextToImage()
        
        generated_images = []
        
        for i, prompt in enumerate(prompt_list):
            try:
                # Use different seed for each image if seed is specified
                current_seed = seed + i if seed != -1 else -1
                
                # Generate single image
                image_tensor = single_node.generate_image(
                    prompt=prompt,
                    api_key=api_key,
                    model=model,
                    size=size,
                    response_format="b64_json",
                    seed=current_seed,
                    guidance_scale=guidance_scale,
                    watermark=watermark
                )[0]
                
                generated_images.append(image_tensor)
                
                # Add small delay between requests to avoid rate limiting
                if i < len(prompt_list) - 1:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Failed to generate image for prompt '{prompt}': {str(e)}")
                continue
        
        if not generated_images:
            raise ValueError("Failed to generate any images")
        
        # Concatenate all images into a single batch tensor
        batch_tensor = torch.cat(generated_images, dim=0)
        
        return (batch_tensor,)

# Additional utility node for generating variations
class SeedreamVariationsTextToImage:
    """
    Generate multiple variations of the same prompt using different seeds
    """
    
    CATEGORY = "image/generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape with mountains and a lake"
                }),
                "model": ([DEFAULT_MODEL], {
                    "default": DEFAULT_MODEL
                }),
                "size": (["1024x1024", "864x1152", "1152x864", "1280x720", "720x1280", "832x1248", "1248x832", "1512x648"], {
                    "default": "1024x1024"
                }),
                "num_variations": ([1, 2, 3, 4], {
                    "default": 2
                }),
                "base_seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1
                }),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {
                    "default": 2.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "watermark": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("variations",)
    FUNCTION = "generate_variations"
    
    def generate_variations(self, prompt, model, size, num_variations, base_seed=-1, guidance_scale=2.5, watermark=True):
        """
        Generate multiple variations of the same prompt using different seeds
        """
        
        # Load API key from environment variable
        api_key = os.getenv('SEEDREAM_API_KEY')
        if not api_key:
            raise ValueError("SEEDREAM_API_KEY not found in environment variables. Please add it to your .env file.")
        
        if not api_key:
            raise ValueError("API key is required")
        
        # Create single node instance for reuse
        single_node = SeedreamTextToImage()
        
        generated_images = []
        
        for i in range(num_variations):
            try:
                # Generate different seeds for variations
                if base_seed == -1:
                    # If no base seed specified, use random seeds
                    current_seed = -1
                else:
                    # Use base_seed + variation index to ensure different but reproducible results
                    current_seed = base_seed + i * 1000  # Multiply by 1000 to ensure significant difference
                    # Ensure seed stays within valid range
                    if current_seed > 2147483647:
                        current_seed = (current_seed % 2147483647) + 1
                
                # Generate single image variation - REMOVED api_key parameter
                image_tensor = single_node.generate_image(
                    prompt=prompt,
                    model=model,
                    size=size,
                    response_format="b64_json",
                    seed=current_seed,
                    guidance_scale=guidance_scale,
                    watermark=watermark
                )[0]
                
                generated_images.append(image_tensor)
                
                # Add small delay between requests to avoid rate limiting
                if i < num_variations - 1:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Failed to generate variation {i+1}: {str(e)}")
                continue
        
        if not generated_images:
            raise ValueError("Failed to generate any image variations")
        
        # Concatenate all images into a single batch tensor
        batch_tensor = torch.cat(generated_images, dim=0)
        
        return (batch_tensor,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "SeedreamTextToImage": SeedreamTextToImage,
    "SeedreamBatchTextToImage": SeedreamBatchTextToImage,
    "SeedreamVariationsTextToImage": SeedreamVariationsTextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedreamTextToImage": "Seedream Text to Image",
    "SeedreamBatchTextToImage": "Seedream Batch Text to Image",
    "SeedreamVariationsTextToImage": "Seedream Variations Text to Image",
}