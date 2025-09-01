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

# Load environment variables
load_dotenv()

# Load model defaults from environment variables
DEFAULT_T2V_MODEL = os.getenv('SEEDANCE_T2V_MODEL', 'seedance-1-0-lite-t2v')
DEFAULT_I2V_MODEL = os.getenv('SEEDANCE_I2V_MODEL', 'seedance-1-0-lite-i2v')

class SeedanceLiteTextToVideo:
    """
    Custom ComfyUI node for ByteDance Seedance-lite text-to-video generation with polling
    """
    
    CATEGORY = "video/generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful sunset over the ocean"
                }),
                "model": ([DEFAULT_T2V_MODEL], {
                    "default": DEFAULT_T2V_MODEL
                }),
            },
            "optional": {
                "resolution": (["480p", "720p", "1080p"], {
                    "default": "720p"
                }),
                "ratio": (["16:9", "9:16", "1:1"], {
                    "default": "16:9"
                }),
                "max_wait_time": ("INT", {
                    "default": 300,
                    "min": 60,
                    "max": 1800
                }),
                "poll_interval": ("INT", {
                    "default": 10,
                    "min": 5,
                    "max": 60
                }),
                "callback_url": ("STRING", {
                    "default": ""
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "status", "task_id")
    FUNCTION = "generate_video_with_polling"
    
    def poll_task_completion(self, task_id, max_wait_time=300, poll_interval=10):
        """Poll the API for task completion"""
        api_key = os.getenv('SEEDANCE_API_KEY')
        if not api_key:
            raise ValueError("SEEDANCE_API_KEY not found in environment variables")
        
        query_url = f"https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(query_url, headers=headers, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                status = result.get("status", "unknown")
                
                if status == "succeeded":
                    video_url = result.get("content", {}).get("video_url", "")
                    return status, video_url, result
                elif status == "failed":
                    error_msg = result.get("error", {}).get("message", "Unknown error")
                    return status, f"Task failed: {error_msg}", result
                elif status in ["queued", "running"]:
                    time.sleep(poll_interval)
                    continue
                else:
                    return status, f"Unknown status: {status}", result
                    
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to query task status: {str(e)}")
        
        return "timeout", "Task timed out", {}
    
    def generate_video_with_polling(self, prompt, model, resolution="720p", ratio="16:9", max_wait_time=300, poll_interval=10, callback_url=""):
        # Load API key from environment variable
        api_key = os.getenv('SEEDANCE_API_KEY')
        if not api_key:
            raise ValueError("SEEDANCE_API_KEY not found in environment variables. Please add it to your .env file.")
        
        # API endpoint
        url = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Build prompt with parameters
        full_prompt = prompt
        if resolution != "720p":
            full_prompt += f" --resolution {resolution}"
        if ratio != "16:9":
            full_prompt += f" --ratio {ratio}"
        
        # Prepare request data
        data = {
            "model": model,
            "content": [
                {
                    "type": "text",
                    "text": full_prompt
                }
            ]
        }
        
        # Add callback URL if provided
        if callback_url:
            data["callback_url"] = callback_url
        
        try:
            # Make API request to create task
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "id" not in result:
                raise ValueError("Video generation failed: No task ID returned from API")
            
            task_id = result["id"]
            
            # Poll for completion
            status, video_url, final_result = self.poll_task_completion(task_id, max_wait_time, poll_interval)
            
            return (video_url, status, task_id)
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Video generation failed: {str(e)}")

class SeedanceLiteImageToVideo:
    """
    Custom ComfyUI node for ByteDance Seedance-lite image-to-video generation with polling
    Supports first frame only or first frame + last frame
    """
    
    CATEGORY = "video/generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_frame": ("IMAGE",),
                "model": ([DEFAULT_I2V_MODEL], {
                    "default": DEFAULT_I2V_MODEL
                }),
            },
            "optional": {
                "last_frame": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "resolution": (["480p", "720p", "1080p"], {
                    "default": "720p"
                }),
                "ratio": (["16:9", "9:16", "1:1"], {
                    "default": "16:9"
                }),
                "duration": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10
                }),
                "framepersecond": ("INT", {
                    "default": 24,
                    "min": 12,
                    "max": 60
                }),
                "watermark": ("BOOLEAN", {
                    "default": False
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
                "camerafixed": ("BOOLEAN", {
                    "default": False
                }),
                "max_wait_time": ("INT", {
                    "default": 300,
                    "min": 60,
                    "max": 1800
                }),
                "poll_interval": ("INT", {
                    "default": 10,
                    "min": 5,
                    "max": 60
                }),
                "callback_url": ("STRING", {
                    "default": ""
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "status", "task_id")
    FUNCTION = "generate_video_from_images_with_polling"
    
    def tensor_to_base64_url(self, tensor):
        """Convert tensor to base64 data URL"""
        # Convert tensor to PIL Image
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert from [0,1] to [0,255] and change to uint8
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        tensor = tensor.clamp(0, 255).byte()
        
        # Convert to numpy array and then PIL Image
        numpy_image = tensor.cpu().numpy()
        pil_image = Image.fromarray(numpy_image, mode='RGB')
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    
    def poll_task_completion(self, task_id, max_wait_time=300, poll_interval=10):
        """Poll the API for task completion"""
        api_key = os.getenv('SEEDANCE_API_KEY')
        if not api_key:
            raise ValueError("SEEDANCE_API_KEY not found in environment variables")
        
        query_url = f"https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(query_url, headers=headers, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                status = result.get("status", "unknown")
                
                if status == "succeeded":
                    video_url = result.get("content", {}).get("video_url", "")
                    return status, video_url, result
                elif status == "failed":
                    error_msg = result.get("error", {}).get("message", "Unknown error")
                    return status, f"Task failed: {error_msg}", result
                elif status in ["queued", "running"]:
                    time.sleep(poll_interval)
                    continue
                else:
                    return status, f"Unknown status: {status}", result
                    
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to query task status: {str(e)}")
        
        return "timeout", "Task timed out", {}
    
    def generate_video_from_images_with_polling(self, first_frame, model, last_frame=None, prompt="", resolution="720p", ratio="16:9", duration=5, framepersecond=24, watermark=False, seed=-1, camerafixed=False, max_wait_time=300, poll_interval=10, callback_url=""):
        # Load API key from environment variable
        api_key = os.getenv('SEEDANCE_API_KEY')
        if not api_key:
            raise ValueError("SEEDANCE_API_KEY not found in environment variables. Please add it to your .env file.")
        
        # API endpoint
        url = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare content array
        content = []
        
        # Add text prompt with parameters first
        if prompt or resolution != "720p" or ratio != "16:9" or duration != 5 or framepersecond != 24 or watermark or seed != -1 or camerafixed:
            full_prompt = prompt
            if resolution != "720p":
                full_prompt += f" --resolution {resolution}"
            if ratio != "16:9":
                full_prompt += f" --ratio {ratio}"
            if duration != 5:
                full_prompt += f" --duration {duration}"
            if framepersecond != 24:
                full_prompt += f" --framepersecond {framepersecond}"
            if watermark:
                full_prompt += " --watermark true"
            if seed != -1:
                full_prompt += f" --seed {seed}"
            if camerafixed:
                full_prompt += " --camerafixed true"
            
            content.append({
                "type": "text",
                "text": full_prompt
            })
        
        # Add first frame image
        first_frame_url = self.tensor_to_base64_url(first_frame)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": first_frame_url
            },
            "role": "first_frame"
        })
        
        # Add last frame if provided
        if last_frame is not None:
            last_frame_url = self.tensor_to_base64_url(last_frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": last_frame_url
                },
                "role": "last_frame"
            })
        
        # Prepare request data
        data = {
            "model": model,
            "content": content
        }
        
        # Add callback URL if provided
        if callback_url:
            data["callback_url"] = callback_url
        
        try:
            # Make API request to create task
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "id" not in result:
                raise ValueError("Video generation failed: No task ID returned from API")
            
            task_id = result["id"]
            
            # Poll for completion
            status, video_url, final_result = self.poll_task_completion(task_id, max_wait_time, poll_interval)
            
            return (video_url, status, task_id)
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Video generation failed: {str(e)}")

class SeedanceLiteReferenceImageToVideo:
    """
    Custom ComfyUI node for ByteDance Seedance-lite reference image-to-video generation with polling
    Supports 1-4 reference images for video generation
    """
    
    CATEGORY = "video/generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",),
                "model": ([DEFAULT_I2V_MODEL], {
                    "default": DEFAULT_I2V_MODEL
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "resolution": (["480p", "720p", "1080p"], {
                    "default": "720p"
                }),
                "ratio": (["16:9", "9:16", "1:1"], {
                    "default": "16:9"
                }),
                "max_wait_time": ("INT", {
                    "default": 300,
                    "min": 60,
                    "max": 1800
                }),
                "poll_interval": ("INT", {
                    "default": 10,
                    "min": 5,
                    "max": 60
                }),
                "callback_url": ("STRING", {
                    "default": ""
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "status", "task_id")
    FUNCTION = "generate_video_from_reference_images_with_polling"
    
    def tensor_to_base64_url(self, tensor):
        """Convert tensor to base64 data URL"""
        # Convert tensor to PIL Image
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert from [0,1] to [0,255] and change to uint8
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        tensor = tensor.clamp(0, 255).byte()
        
        # Convert to numpy array and then PIL Image
        numpy_image = tensor.cpu().numpy()
        pil_image = Image.fromarray(numpy_image, mode='RGB')
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    
    def poll_task_completion(self, task_id, max_wait_time=300, poll_interval=10):
        """Poll the API for task completion"""
        api_key = os.getenv('SEEDANCE_API_KEY')
        if not api_key:
            raise ValueError("SEEDANCE_API_KEY not found in environment variables")
        
        query_url = f"https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(query_url, headers=headers, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                status = result.get("status", "unknown")
                
                if status == "succeeded":
                    video_url = result.get("content", {}).get("video_url", "")
                    return status, video_url, result
                elif status == "failed":
                    error_msg = result.get("error", {}).get("message", "Unknown error")
                    return status, f"Task failed: {error_msg}", result
                elif status in ["queued", "running"]:
                    time.sleep(poll_interval)
                    continue
                else:
                    return status, f"Unknown status: {status}", result
                    
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to query task status: {str(e)}")
        
        return "timeout", "Task timed out", {}
    
    def generate_video_from_reference_images_with_polling(self, reference_images, model, prompt="", resolution="720p", ratio="16:9", max_wait_time=300, poll_interval=10, callback_url=""):
        # Load API key from environment variable
        api_key = os.getenv('SEEDANCE_API_KEY')
        if not api_key:
            raise ValueError("SEEDANCE_API_KEY not found in environment variables. Please add it to your .env file.")
        
        # API endpoint
        url = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare content array
        content = []
        
        # Add text prompt with parameters first
        if prompt or resolution != "720p" or ratio != "16:9":
            full_prompt = prompt
            if resolution != "720p":
                full_prompt += f" --resolution {resolution}"
            if ratio != "16:9":
                full_prompt += f" --ratio {ratio}"
            
            content.append({
                "type": "text",
                "text": full_prompt
            })
        
        # Handle batch of images (up to 4 reference images)
        # reference_images is a batch tensor with shape [batch_size, height, width, channels]
        batch_size = reference_images.shape[0]
        max_images = min(batch_size, 4)  # Limit to 4 images as per API specification
        
        for i in range(max_images):
            image_tensor = reference_images[i]
            image_url = self.tensor_to_base64_url(image_tensor)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
                "role": "reference_image"
            })
        
        # Prepare request data
        data = {
            "model": model,
            "content": content
        }
        
        # Add callback URL if provided
        if callback_url:
            data["callback_url"] = callback_url
        
        try:
            # Make API request to create task
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "id" not in result:
                raise ValueError("Video generation failed: No task ID returned from API")
            
            task_id = result["id"]
            
            # Poll for completion
            status, video_url, final_result = self.poll_task_completion(task_id, max_wait_time, poll_interval)
            
            return (video_url, status, task_id)
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Video generation failed: {str(e)}")

# Node mappings
NODE_CLASS_MAPPINGS = {
    "SeedanceLiteTextToVideo": SeedanceLiteTextToVideo,
    "SeedanceLiteImageToVideo": SeedanceLiteImageToVideo,
    "SeedanceLiteReferenceImageToVideo": SeedanceLiteReferenceImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedanceLiteTextToVideo": "Seedance-lite Text to Video (Complete)",
    "SeedanceLiteImageToVideo": "Seedance-lite Image to Video (Complete)",
    "SeedanceLiteReferenceImageToVideo": "Seedance-lite Reference Images to Video (Complete)",
}