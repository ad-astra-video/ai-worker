import asyncio
import os
import json
from typing import Dict
import logging
import torch
import time

from app.pipelines.base import Pipeline
from app.pipelines.utils import (
    SafetyChecker,
)
from app.pipelines.backends.comfyui import ComfyUIBackend

logger = logging.getLogger(__name__)

class BatchPipeline(Pipeline):
    def __init__(self, **kwargs):
        safety_checker_device = os.getenv("SAFETY_CHECKER_DEVICE", "cpu").lower()
        self._safety_checker = SafetyChecker(device=safety_checker_device)
        #setup tracking for allocation GPUs to pipelines
        self.gpus_lock = asyncio.Lock()
        self.gpu_device_locks = {i: asyncio.Lock() for i in range(torch.cuda.device_count())}
        self.gpus = [i for i in range(torch.cuda.device_count())]
        self.pipeline_gpus = {}
        self.pipeline_last_used = {}
        self.pipeline_restrictions = {}

        #setup backends and pipelines
        self.backends = {}
        self.pipelines_backends = {}
        pipelines_path = "/app/settings/pipelines"
        for filename in os.listdir(pipelines_path):
            if filename.endswith('.json'):
                backend, pipeline_id = filename.replace(".json","").split("--", 1)
                with open(os.path.join(pipelines_path, filename), 'r') as f:
                    try:
                        pipeline_settings = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to get pipeline settings, JSON is invalid {filename}: {e}")
                        continue
                    
                    if "restrictions" in pipeline_settings:
                        self.pipeline_restrictions[pipeline_id] = pipeline_settings["restrictions"]
                #add new backend types here
                if backend == "comfyui":
                    self.pipelines_backends[pipeline_id] = "comfyui"
                    if not "comfyui" in self.backends:
                        self.backends["comfyui"] = ComfyUIBackend()
                    

    async def __call__(self, pipeline_name, model_id, params: Dict[str, any], files: Dict[str, any], **kwargs):
        pipeline_id = f"{pipeline_name}--{model_id.replace('/', '--')}"
        backend = self.pipelines_backends.get(pipeline_id, "")
        if backend == "":
            logger.error(f"No backend found for pipeline {pipeline_id}. pipelines_backends: {self.pipelines_backends}")
            raise ValueError(f"No backend found for pipeline {pipeline_id}.")

        # reuse the GPU if the pipeline is already running
        cuda_device = -1
        if pipeline_id in self.pipeline_gpus:
            cuda_device = self.pipeline_gpus[pipeline_id]
            logger.info(f"Pipeline {pipeline_id} is already running on GPU {cuda_device}. Reusing the GPU.")
        else:
            # if the pipeline is not running, allocate a GPU
            async with self.gpus_lock:
                if len(self.gpus) > 0:
                    #if pipeline has restrictions, check if we can use the GPU
                    for device in self.gpus:
                        if await self.passes_restrictions(self.pipeline_restrictions.get(pipeline_id, {}), device):
                            cuda_device = device
                            self.gpus.remove(cuda_device)
                            break
                        else:
                            #no restrictions, just use the first available GPU
                            cuda_device = self.gpus.pop(0)
                    
                    self.pipeline_gpus[pipeline_id] = cuda_device
                else:
                    #stop the pipeline longest not used and use that GPU
                    sorted_pipelines = sorted(self.pipeline_last_used, key=self.pipeline_last_used.get)
                    
                    for running_pipeline_id in sorted_pipelines:
                        cuda_device = self.pipeline_gpus[running_pipeline_id]
                        if await self.passes_restrictions(self.pipeline_restrictions.get(running_pipeline_id, {}), cuda_device):
                            async with self.gpu_device_locks[cuda_device]:
                                await self.backends[backend].stop_pipeline(running_pipeline_id)
                                self.pipeline_gpus[running_pipeline_id] = cuda_device
                                break

                if cuda_device == -1:
                    logger.info(f"Pipeline {pipeline_id} has no available GPUs that pass the restrictions, trying to stop a pipeline to free up a GPU.")
                    return None
                #track last used time for the pipeline
                self.pipeline_last_used[pipeline_id] = time.time()
                
        #get the GPU device lock for the allocated GPU, release it after processing
        start = time.time()
        async with self.gpu_device_locks[cuda_device]:
            logger.info(f"Processing pipeline {pipeline_id} on GPU {cuda_device} with backend {backend}   waited={round(time.time()-start,2)}seconds.")
            result = await self.backends[backend].process(cuda_device, pipeline_id, params, files, **kwargs)

            if "safety_check" in kwargs:
                if "images" in result:
                    images, nsfws = self._safety_checker.check_nsfw_images(result["images"])
                    for i, _ in enumerate(images):
                        result["images"][i]["nsfw"] = nsfws[i]
            
            if result is None:
                raise ValueError("No result returned from backend.")
            
            return result

    async def passes_restrictions(self, restrictions: Dict[str, any], cuda_device: int) -> bool:
        """
        Check if the GPU passes the restrictions for the pipeline.
        """
        if "minimum_vram" in restrictions:
            min_vram = float(restrictions["minimum_vram"].upper().replace("GB", "").strip())
            device_vram = torch.cuda.get_device_properties(cuda_device).total_memory / (1024 ** 3)
            if device_vram < min_vram:
                return False

        #return True if all restrictions are passed
        return True

    async def get_pipelines(self):
        """
        Get the list of pipelines.
        """
        logger.info("Getting pipelines advertising info...")
        pipelines = []
        pipelines_path = "/app/settings/pipelines"
        for filename in os.listdir(pipelines_path):
            if filename.endswith('.json'):
                backend, pipeline = filename.split("--", 1)
                pipeline = pipeline.replace(".json", "")
                pipeline_name, model_id = pipeline.split("--", 1)
                model_id = model_id.replace("--", "/")
                pipeline_json = {
                    "pipeline": pipeline_name,
                    "model_id": model_id,
                    "capacity": 1,
                }

                with open(os.path.join(pipelines_path, filename), 'r') as f:
                    try:
                        pipeline_settings = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to get pipeline settings, JSON is invalid {filename}: {e}")
                        
                    if "pricing" in pipeline_settings:
                        if "price_per_unit" in pipeline_settings["pricing"]:
                            pipeline_json["price_per_unit"] = pipeline_settings["pricing"]["price_per_unit"]
                        if "currency" in pipeline_settings["pricing"]:
                            pipeline_json["currency"] = pipeline_settings["pricing"]["currency"]
                        if "price_scaling" in pipeline_settings["pricing"]:
                            pipeline_json["price_scaling"] = pipeline_settings["pricing"]["price_scaling"]
                        else:
                            pipeline_json["price_scaling"] = 1
                        
                    pipelines.append(pipeline_json)
        
        return pipelines
            
    async def refresh_pipelines(self):
        """
        Refresh the list of pipelines.
        """
        for backend in self.backends:
            asyncio.to_thread(self.backend.setup_pipelines)
        
    