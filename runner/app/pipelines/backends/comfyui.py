import json
import os
import logging
import time
from git import Repo
from app.pipelines.utils import resumable_download
from app.pipelines.base import Backend
import re
from typing import Dict
import os
import httpx
import asyncio
from app.routes.utils import image_to_data_url, audio_to_data_url
from app.pipelines.backends.utils import run_command, start_backend, stop_backend
import io
import sys
import threading
from PIL import Image
import torch

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.FATAL)
class ComfyUIBackend(Backend):
    def __init__(self):
        self.pipelines_lock = threading.Lock()
        self.pipelines = {}
        self.pipeline_venvs = {}
        #check if comfyui is installed
        self.setup_pipelines()

    def _create_pipeline_env(self, pipeline):
        pyenv_root = os.getenv("PYENV_ROOT", "/root/.pyenv")
        logger.info(f"Creating virtualenv for {pipeline}")
        run_command(f"python -m venv /app/virtual_envs/{pipeline}")
        run_command(f"ln -s /app/virtual_envs/{pipeline} {pyenv_root}/versions/{pipeline}")
        #run_command(f"pyenv virtualenv --system-site-packages /app/virtual_envs/{pipeline}")
        run_command(f"pyenv local {pipeline}")
        logger.info(f"Created virtualenv {pipeline}")
        logger.info(f"Installing pytorch and dependencies for {pipeline}")
        run_command(f"pyenv activate {pipeline} && pip install torch==2.7 torchvision torchaudio transformers diffusers numpy>=1.26.4 accelerate tqdm")
        logger.info(f"Installed torch and dependencies for {pipeline}")
        logger.info(f"Installing comfyui dependencies for {pipeline}")
        run_command(f"pyenv activate {pipeline} && pip install -r /app/workspace/requirements.txt")
        logger.info(f"Installed comfyui dependencies for {pipeline}")
        #logger.info(f"Installing comfyui manager dependencies for {pipeline}")
        #run_command(f"pyenv activate {pipeline} && pip install -r /app/workspace/custom_nodes/ComfyUI-Manager/requirements.txt")
        #logger.info(f"Installed comfyui manager dependencies for {pipeline}")
    
    def _install_node(self, node, pipeline):
        if 'url' in node:
            custom_nodes_path = "/app/workspace/custom_nodes"
            node_url = node['url']
            node_name = node['name']
            node_path = os.path.join(custom_nodes_path, node_name)
            if not os.path.exists(node_path):
                logger.debug(f"Installing node {node_name} from {node_url} to {node_path}")
                repo = Repo.clone_from(node_url, node_path)
                if "branch" in node:
                    repo.git.checkout(node["branch"])
                if os.path.exists(node_path+"/requirements.txt"):
                    output = run_command(f"pyenv activate {pipeline} && pip install -r {node_path}/requirements.txt")
                    logger.debug(output)
                logger.info(f"Node {node_name} installed at {node_path}")
            else:
                if "auto_update" in node and node["auto_update"]:
                    logger.debug(f"Updating node {node_name} from {node_url} to {node_path}")
                    repo = Repo(node_path)
                    repo.git.reset('--hard')
                    for remote in repo.remotes:
                        remote.fetch()
                    if "branch" in node:
                        repo.git.checkout(node["branch"])
                    else:
                        repo.git.checkout("main")

                    if os.path.exists(node_path+"/requirements.txt"):
                        output = run_command(f"pyenv activate {pipeline} && pip install -r {node_path}/requirements.txt")
                        logger.debug(output)
                    logger.debug(f"Node {node_name} updated at {node_path}")
                else:
                    logger.debug(f"Node {node_name} already installed at {node_path} (auto_update not enabled)")
        if 'addl_requirements' in node:
            if node["addl_requirements"] != "":
                requirements_list = node['addl_requirements'].split(",")
                for req in requirements_list:
                    logger.debug(f"Installing additional requirements {req} for node {node_name}")
                    run_command(f"pyenv activate {pipeline} && pip install {req}")
            
    def _download_model(self, model_url, model_path):
        file_path = "/app/workspace/"+model_path
        # Check if the model is already downloaded
        if os.path.exists(file_path):
            logger.debug(f"File {file_path} already exists, skipping download.")
            return
        logger.debug(f"Downloading {model_url} → {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        #download the model
        resumable_download(model_url, file_path)
   
    def _update_prompt_fields(self, prompt: str, data: dict) -> str:
        return re.sub(r"\|([^|]+)\|", lambda m: str(data.get(m.group(1), m.group(0))), json.dumps(prompt))
    
    def _extract_seed_from_prompt(self, prompt: str) -> int:
        # Extract the seed from the prompt using regex
        match = re.search(r'("(?:noise_)?seed": )(\d+)', prompt)
        if match:
            return int(match.group(2))
        return None
    
    async def _download_result(self, url: str) -> bytes:
        # Download the result from the given URL
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code == 200:
                return response.content
            else:
                raise ValueError(f"Failed to download result: {response.status_code} {response.text}")
            
    def setup_pipelines(self):
        self.pipelines_lock.acquire()
        logger.info("Setting up pipelines...")
        
        pipelines_path = "/app/settings/pipelines"
        pipelines_files = {}
        for filename in os.listdir(pipelines_path):
            if filename.endswith('.json') and filename.startswith("comfyui--"):
                pipeline = filename.replace("comfyui--", "").replace(".json", "")
                logger.info(f"found pipeline {pipeline}, setting up nodes and downloading models as needed")
                with open(os.path.join(pipelines_path, filename), 'r') as f:
                    try:
                        pipeline_settings = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to get pipeline settings, JSON is invalid {filename}: {e}")
                        
                    pipelines_files[pipeline] = pipeline_settings

        for pipeline in pipelines_files:
            if pipeline in self.pipelines:
                continue

            pipeline_settings = pipelines_files[pipeline] 
            #------------------------------
            # install nodes for the pipeline and set venv mapping
            #------------------------------ 
            pipeline_venv = "comfyui-base"
            if 'nodes' in pipeline_settings:
                #install the nodes in separate virtualenvs
                pipeline_venv = pipeline
                try:
                    #create the virtualenv if needed
                    virtualenv_root = os.getenv("PYENV_ROOT", "/root/.pyenv")
                    if not os.path.exists(f"{virtualenv_root}/versions/{pipeline}"):
                        self._create_pipeline_env(pipeline)
                except Exception as e:
                    logger.error(f"Failed to create virtualenv for {pipeline}: {e}")
                    raise ValueError(f"Failed to create virtualenv for {pipeline}.")
                #install the nodes and additional requirements
                for node in pipeline_settings['nodes']:
                    self._install_node(node, pipeline)
            
            self.pipeline_venvs[pipeline] = pipeline_venv

            #------------------------------
            # download models
            #------------------------------       
            if 'models' in pipeline_settings:
                for model in pipeline_settings['models']:
                    model_path = model
                    model_url = pipeline_settings['models'][model]
                    self._download_model(model_url, model_path)
            
            #------------------------------
            #download static inputs to workflow if needed
            #------------------------------
            if "inputs" in pipeline_settings:
                for input in pipeline_settings["inputs"]:
                    input_save_path = "/app/workspace/input"
                    if "subfolder" in pipeline_settings["inputs"][input]:
                        input_save_path = os.path.join(input_save_path, pipeline_settings["inputs"][input]["subfolder"])
                    input_save_path = os.path.join(input_save_path, pipeline_settings["inputs"][input]["name"])
                
                    self._download_model(pipeline_settings["inputs"][input]["url"], input_save_path)

        #release lock
        self.pipelines_lock.release()

    async def stop_pipeline(self, pipeline_runner_id: str):
        start = time.time()
        logger.info(f"Stopping pipeline {pipeline_runner_id}...")
        #stop the backend
        stop_backend(pipeline_runner_id)
        logger.info(f"Pipeline {pipeline_runner_id} stopped took={time.time() - start:.2f}seconds")

    async def process(self, cuda_device: int, port: str, pipeline_id: str, params: Dict[str, any], files: Dict[str, any]):
        logger.info(f"ComfyUI workflow proxying for path: {pipeline_id}")
        
        pipeline_settings_path = f"/app/settings/pipelines/comfyui--{pipeline_id}.json"
        if not os.path.exists(pipeline_settings_path):
            raise ValueError(f"Pipeline settings file not found: {pipeline_settings_path}")
        
        pipeline_settings = {}
        with open(pipeline_settings_path, "r") as f:
            pipeline_settings = json.load(f)
            if not "prompt" in pipeline_settings:
                raise ValueError(f"Prompt not found in pipeline settings: {pipeline_settings_path}")
                
        backend_url = f"http://localhost:{port}"
        
        #client to send data to backend
        client = httpx.AsyncClient()
        backend_running = False
        #check if backend is up, and start if not
        try:
            resp = await client.get(f"{backend_url}/system_stats", timeout=.5)
            if resp.status_code == 200:
                logger.info(f"Backend already running for {pipeline_id}, using existing backend")
                backend_running = True
        except httpx.ConnectError as e:
            logger.info(f"Backend not running, starting backend for {pipeline_id}: {e}")
        #if not running, start the backend
        if not backend_running:
            venv_name = self.pipeline_venvs[pipeline_id]
            start_backend(pipeline_id, port, cuda_device, venv_name)

            #wait for startup
            while True:
                await asyncio.sleep(1)
                try:
                    resp = await client.get(f"{backend_url}/system_stats", timeout=2)
                    if resp.status_code == 200:
                        break
                    logger.info("waiting for comfyui to startup...")
                except Exception as e:
                    logger.error(f"Error connecting to ComfyUI backend: {e}")

        #upload the files and add to the prompt
        inputs_to_remove = []
        for file in files:
            logger.info(f"Uploading file {file} to backend")
            filename, file_content, content_type = files[file]

            upload_data = {
                "overwrite": "true",
                "type": "input"
            }
            
            if content_type == "image/mask":
                upload_files = {"image": (filename, file_content, "image/png")}
                resp = await client.post(
                    url=f"{backend_url}/upload/mask",
                    data=upload_data,
                    files=upload_files,
                )

                if resp.status_code != 200:
                    raise ValueError(f"Failed to upload mask: {resp.text}")
            elif "image/" in content_type or file == "image":
                upload_files = {"image": (filename, file_content, "image/png")}
                resp = await client.post(
                    url=f"{backend_url}/upload/image",
                    data=upload_data,
                    files=upload_files,
                )

                if resp.status_code != 200:
                    raise ValueError(f"Failed to upload image: {resp.text}")
                        
            result = resp.json()
            if 'subfolder' in result and result['subfolder']:
                inputs_to_remove.append(f"/app/workspace/{result['type']}/{result['subfolder']}/{result['name']}")
            else:
                inputs_to_remove.append(f"/app/workspace/{result['type']}/{result['name']}")
            
            #update the prompt with the uploaded file path
            params[file] = result["name"]
            
        #add defaults if needed
        if "defaults" in pipeline_settings:
            for input in pipeline_settings["defaults"]:
                if not input in params:
                    params[input] = pipeline_settings["defaults"][input]

        #update the prompt for processing
        prompt = json.dumps(pipeline_settings["prompt"])
        prompt_with_data = self._update_prompt_fields(prompt, params)
        logger.info(f"Prompt with data: {prompt_with_data}")
        
        #queue the prompt
        resp = await client.post(
            url=f"{backend_url}/prompt",
            json={"prompt":json.loads(prompt_with_data)},
            timeout=None  # Optional: disable timeout for SSE
        )

        logger.info(f"Prompt queued")
        if resp.status_code != 200:
            raise ValueError(f"Failed to queue prompt: {resp.text}")

        prompt_id = resp.json()
        prompt_id = prompt_id["prompt_id"]
        logger.info(f"Prompt ID: {prompt_id}")
        status_json = {}
        while True:
            await asyncio.sleep(0.5)
            status = await client.get(
                url=f"{backend_url}/history/{prompt_id}",
                timeout=None  # Optional: disable timeout for SSE
            )

            if status.status_code != 200:
                break
            
            status_json = status.json()
            #no status available, continue
            if not prompt_id in status_json:
                continue
            prompt_result = status_json[prompt_id]

            if 'status' in prompt_result:
                prompt_result_status = prompt_result['status']
                if prompt_result_status['completed']:
                    if prompt_result_status['status_str'] == "success":
                        logger.info(f"Prompt processed successfully: {prompt_id}")
                    else:
                        logger.error(f"Prompt processing failed: {prompt_id} {prompt_result_status}")
                        raise ValueError(f"Prompt processing failed: {prompt_result_status}")
                
                break

            logger.info(f"Waiting for prompt to be processed: {prompt_id}")
            
        
        #download the output files and return
        combined_output = []
        outputs = prompt_result.get("outputs", {})
        logger.debug(f"Outputs: {outputs}")
        seed = self._extract_seed_from_prompt(json.dumps(prompt_result.get("prompt", {})))
        output_type = ""
        for output_node in outputs:
            for output_type in outputs[output_node]:
                if output_type == "images" or output_type == "audio":
                    for result in outputs[output_node][output_type]:
                        result = await client.get(f"{backend_url}/view?filename={result['filename']}&subfolder={result['subfolder']}&type={result['type']}")
                        if result.status_code != 200:
                            raise ValueError(f"Failed to download result: {result.text}")
                        
                        result_bytes = io.BytesIO(result.content)
                        if output_type == "images":
                            pil_img = Image.open(result_bytes)
                            combined_output.append({"url": image_to_data_url(pil_img), "seed": seed})
                        elif output_type == "audio":
                            combined_output.append({"url": audio_to_data_url(result_bytes)})
                        elif output_type == "json":
                            combined_output.append({"text": result.content})        
        try:
            #remove the input files from the workspace
            for input_file in inputs_to_remove:
                if os.path.exists(input_file):
                    os.remove(input_file)
        except Exception as e:
            logger.error(f"Failed to remove input file(s): {e}")

        return {output_type: combined_output}
        


    