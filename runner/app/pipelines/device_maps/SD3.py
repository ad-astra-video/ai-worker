

import inspect
import logging
import os
import time
from typing import List, Optional, Tuple

import PIL
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines import ImagePipelineOutput

from app.pipelines.base import Pipeline

logger = logging.getLogger(__name__)

class LPStableDiffusion3Pipeline(Pipeline):
    """custom device map for Stable Diffusion 3.x pipelines to run on consumer GPUs"""
    class_vars = ['ldm', 'ldm2', 'lp_device_map']

    def __init__(self, model_id: str, device_map: str, torch_device: any, **kwargs):
        self.lp_device_map = device_map

        if self.lp_device_map == "SD3_DEVICE_MAP_2_GPU":
            #setup transformer for GPU 0
            self.ldm = StableDiffusion3Pipeline.from_pretrained(model_id, text_encoder=None, text_encoder_2=None, text_encoder_3=None, tokenizer=None, tokenizer_2=None, vae=None, **kwargs).to("cuda:0")
            #setup pipeline for all other components on GPU 1
            self.ldm2 = StableDiffusion3Pipeline.from_pretrained(model_id, unet=None, transformer=None, **kwargs).to("cuda:1")
        elif self.device_map == "SD3_DEVICE_MAP_1_GPU":
            self.ldm = StableDiffusion3Pipeline.from_pretrained(model_id, **kwargs)
            self.ldm.enable_model_cpu_offload()
        elif "device_map" in kwargs:
            self.ldm = StableDiffusion3Pipeline.from_pretrained(model_id, **kwargs)
        else:
            self.ldm = self.ldm = StableDiffusion3Pipeline.from_pretrained(model_id, **kwargs).to(torch_device) 
    
    def __getattr__(self, name):
        # Redirect attribute access to self.ldm if it exists there
        try:
            if name not in self.class_vars:
                return getattr(self.ldm, name)
            else:
                return super().__getattr__(name)
            
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        # Handle setting attributes
        if name not in self.class_vars:
            # Redirect to ldm if attribute doesn't exist in this instance
            setattr(self.ldm, name, value)
        else:
            super().__setattr__(name, value)

    def __call__(
        self, prompt: str, **kwargs
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:
        outputs = None
        if self.lp_device_map == "SD3_DEVICE_MAP_2_GPU":
            with torch.no_grad():
                #generate prompt embeddings on GPU 1
                start = time.time()
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = None, None, None, None
                encode_prompt_kwargs = inspect.signature(self.ldm2.encode_prompt).parameters.keys()
                prompt_2 = kwargs.pop("prompt_2", "")
                prompt_3 = kwargs.pop("prompt_3", "")
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.ldm2.encode_prompt(prompt, prompt_2, prompt_3, **{k: v for k, v in kwargs.items() if k in encode_prompt_kwargs})
                logger.info(f"encode_prompt took: {time.time()-start} seconds")
                #generate the image with transformer, return latents
                start = time.time()
                prompt_embeds = prompt_embeds.to(self.ldm._execution_device)
                negative_prompt_embeds = negative_prompt_embeds.to(self.ldm._execution_device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(self.ldm._execution_device)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(self.ldm._execution_device)
                logger.info(f"prompt embeds conversion took: {time.time()-start} seconds")
                start= time.time()
                ldm_kwargs = inspect.signature(self.ldm.__call__).parameters.keys()
                latents = self.ldm(prompt=None, prompt_2=None, 
                                prompt_embeds=prompt_embeds, 
                                pooled_prompt_embeds=pooled_prompt_embeds, 
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                output_type="latent", return_dict=False,
                                **{k: v for k, v in kwargs.items() if k in ldm_kwargs})
                logger.info(f"transformer took: {time.time()-start} seconds")
                #use the VAE on GPU 1 to process the image
                #copied from diffusers/pipelines/flux/pipeline_flux.py L760
                start = time.time()
                latents = latents[0].to(self.ldm2._execution_device)
                logger.info(f"latents conversion took: {time.time()-start} seconds")
                start = time.time()
                
                latents = (latents / self.ldm2.vae.config.scaling_factor) + self.ldm2.vae.config.shift_factor
                image = self.ldm2.vae.decode(latents, return_dict=False)[0]
                image = self.ldm2.image_processor.postprocess(image) #only support default output_type="pil"
                logger.info(f"vae decode took: {time.time()-start} seconds")
                
                outputs = ImagePipelineOutput(images=image)                
        else:
            outputs = self.ldm(prompt=prompt, **kwargs)

        return outputs