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
DEFAULT_MODEL = os.getenv('SEEDEDIT_MODEL', 'seededit-3-0-i2i-250628')

class SeedEditImageToImage:
    """
    Custom ComfyUI node for ByteDance SeedEdit image-to-image editing
    """
    
    CATEGORY = "image/editing"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Make the image more vibrant and colorful"
                }),
                "model": ([DEFAULT_MODEL], {
                    "default": DEFAULT_MODEL
                }),
                "response_format": (["url", "b64_json"], {
                    "default": "b64_json"
                }),
                "size": (["adaptive"], {
                    "default": "adaptive"
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
                    "default": 5.5,
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
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    
    def tensor_to_base64(self, tensor):
        """
        Convert ComfyUI image tensor to base64 string
        """
        # Convert tensor to PIL Image
        # ComfyUI tensors are in format [B, H, W, C] with values in range [0, 1]
        if len(tensor.shape) == 4:
            # Take first image from batch
            tensor = tensor[0]
        
        # Convert to numpy array and scale to 0-255
        image_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array, mode='RGB')
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"
    
    def edit_image(self, image, prompt, model, response_format, size, seed=-1, guidance_scale=5.5, watermark=True):
        """
        Edit image using BytePlus SeedEdit API
        """
        
        # Load API key from environment variable
        api_key = os.getenv('SEEDEDIT_API_KEY')
        if not api_key:
            raise ValueError("SEEDEDIT_API_KEY not found in environment variables. Please add it to your .env file.")
        
        # API endpoint
        url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Convert input image tensor to base64
        image_base64 = self.tensor_to_base64(image)
        
        # Prepare request data
        data = {
            "model": model,
            "prompt": prompt,
            "image": image_base64,
            "response_format": response_format,
            "size": size,
            "guidance_scale": guidance_scale,
            "watermark": watermark
        }
        
        # Add seed if specified
        if seed != -1:
            data["seed"] = seed
        
        try:
            # Make API request
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            if "data" not in result or not result["data"]:
                raise ValueError("No image data returned from API")
            
            # Process the response based on format
            if response_format == "b64_json":
                # Decode base64 image
                image_data = result["data"][0]["b64_json"]
                image_bytes = base64.b64decode(image_data)
                edited_image = Image.open(BytesIO(image_bytes))
            else:
                # Download image from URL
                image_url = result["data"][0]["url"]
                img_response = requests.get(image_url, timeout=30)
                img_response.raise_for_status()
                edited_image = Image.open(BytesIO(img_response.content))
            
            # Convert PIL image to tensor format expected by ComfyUI
            # ComfyUI expects images in format [B, H, W, C] with values in range [0, 1]
            edited_image = edited_image.convert("RGB")
            image_array = torch.from_numpy(np.array(edited_image)).float() / 255.0
            
            # Add batch dimension
            image_tensor = image_array.unsqueeze(0)
            
            return (image_tensor,)
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Image editing failed: {str(e)}")

# Additional utility node for batch editing
class SeedEditBatchImageToImage:
    """
    Batch version of SeedEdit image-to-image editing
    """
    
    CATEGORY = "image/editing"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Make the image more vibrant and colorful"
                }),
                "model": ([DEFAULT_MODEL], {
                    "default": DEFAULT_MODEL
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
                    "default": 5.5,
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
    RETURN_NAMES = ("edited_images",)
    FUNCTION = "edit_batch_images"
    
    def edit_batch_images(self, images, prompt, model, seed=-1, guidance_scale=5.5, watermark=True):
        """
        Edit multiple images with the same prompt
        """
        
        # Load API key from environment variable
        api_key = os.getenv('SEEDEDIT_API_KEY')
        if not api_key:
            raise ValueError("SEEDEDIT_API_KEY not found in environment variables. Please add it to your .env file.")
        
        # Create single node instance for reuse
        single_node = SeedEditImageToImage()
        
        edited_images = []
        
        # Process each image in the batch
        for i in range(images.shape[0]):
            try:
                # Extract single image from batch
                single_image = images[i:i+1]  # Keep batch dimension
                
                # Use different seed for each image if seed is specified
                current_seed = seed + i if seed != -1 else -1
                
                # Edit single image
                edited_tensor = single_node.edit_image(
                    image=single_image,
                    prompt=prompt,
                    model=model,
                    response_format="b64_json",
                    size="adaptive",
                    seed=current_seed,
                    guidance_scale=guidance_scale,
                    watermark=watermark
                )[0]
                
                edited_images.append(edited_tensor)
                
                # Add small delay between requests to avoid rate limiting
                if i < images.shape[0] - 1:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Failed to edit image {i+1}: {str(e)}")
                continue
        
        if not edited_images:
            raise ValueError("Failed to edit any images")
        
        # Concatenate all images into a single batch tensor
        batch_tensor = torch.cat(edited_images, dim=0)
        
        return (batch_tensor,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "SeedEditImageToImage": SeedEditImageToImage,
    "SeedEditBatchImageToImage": SeedEditBatchImageToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedEditImageToImage": "SeedEdit Image to Image",
    "SeedEditBatchImageToImage": "SeedEdit Batch Image to Image",
}