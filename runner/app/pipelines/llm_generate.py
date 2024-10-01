import asyncio
import logging
import os
import psutil
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from huggingface_hub import file_download, snapshot_download, hf_hub_download
from threading import Thread

logger = logging.getLogger(__name__)

def get_max_memory():
    num_gpus = torch.cuda.device_count()
    gpu_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory // 1024**3}GiB" for i in range(num_gpus)}
    cpu_memory = f"{psutil.virtual_memory().available // 1024**3}GiB"
    max_memory = {**gpu_memory, "cpu": cpu_memory}
    
    logger.info(f"Max memory configuration: {max_memory}")
    return max_memory

def load_model_8bit(model_id: str, **kwargs):
    max_memory = get_max_memory()

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory,
        offload_folder="offload",
        low_cpu_mem_usage=True,
        **kwargs
    )

    return tokenizer, model

def load_model_fp16(model_id: str, **kwargs):
    device = get_torch_device()
    max_memory = get_max_memory()
    
    # Check for fp16 variant
    local_model_path = os.path.join(get_model_dir(), file_download.repo_folder_name(repo_id=model_id, repo_type="model"))
    has_fp16_variant = any(".fp16.safetensors" in fname for _, _, files in os.walk(local_model_path) for fname in files)
    
    if device != "cpu" and has_fp16_variant:
        logger.info("Loading fp16 variant for %s", model_id)
        kwargs["torch_dtype"] = torch.float16
        kwargs["variant"] = "fp16"
    elif device != "cpu":
        kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    
    config = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).config
    
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    checkpoint_dir = snapshot_download(model_id, cache_dir=get_model_dir(), local_files_only=True)
    
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint_dir,
        device_map="auto",
        max_memory=max_memory,
        no_split_module_classes=["LlamaDecoderLayer"],  # Adjust based on your model architecture
        dtype=kwargs.get("torch_dtype", torch.float32),
        offload_folder="offload",
        offload_state_dict=True,
    )

    return tokenizer, model

def load_with_exllamav2(model_id: str, **kwargs):
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache_Q4, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2DynamicGeneratorAsync

    device = get_torch_device()
    # Check for fp16 variant
    config_file_path = hf_hub_download(model_id, filename="config.json", cache_dir=get_model_dir())
    local_model_path = os.path.dirname(config_file_path)
    quantized_path = local_model_path+"/quantized-exl2/6bpw"
    os.makedirs(quantized_path, exist_ok=True) #make sure output directory exists

    logger.info(f"loading model with exllamma2 from model_path={local_model_path} with quantized_model_path={quantized_path}")
    
    config = ExLlamaV2Config(quantized_path)
    config.arch_compat_overrides()

    config.max_input_len = 8 * 1024 #8k context window
    max_batch_size = 5
    if os.getenv("MAX_BATCH_SIZE", "") != "":
        max_batch_size = int(os.getenv("MAX_BATCH_SIZE"))
    if os.getenv("MAX_INPUT_LEN", "") != "":
        config.max_input_len = int(os.getenv("MAX_INPUT_LEN"))
    max_seq_len = config.max_input_len * 5
    if os.getenv("MAX_SEQ_LEN","") != "":
        max_seq_len = int(os.getenv("MAX_SEQ_LEN"))
   
    config.max_attention_size = config.max_input_len ** 2
    model = ExLlamaV2(config)
    
    cache = ExLlamaV2Cache_Q4(model, max_seq_len=max_seq_len, lazy=True)
        
    model.load_autosplit(cache, progress = True)
    tokenizer = ExLlamaV2Tokenizer(config)
    
    generator = ExLlamaV2DynamicGeneratorAsync(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
        max_batch_size = max_batch_size,
    )
    
    logger.info("model loaded, ready to generate")
    
    return tokenizer, model, config, cache, generator

    
class LLMGeneratePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {
            "cache_dir": get_model_dir(),
            "local_files_only": True,
        }
        self.device = get_torch_device()

        if os.getenv("USE_EXL2", "") != "":
            self.tokenizer, self.model, self.config, self.cache, self.generator = load_with_exllamav2(model_id, **kwargs)
        else: 
            # Generate the correct folder name
            folder_path = file_download.repo_folder_name(repo_id=model_id, repo_type="model")
            self.local_model_path = os.path.join(get_model_dir(), folder_path)
            self.checkpoint_dir = snapshot_download(model_id, cache_dir=get_model_dir(), local_files_only=True)

            logger.info(f"Local model path: {self.local_model_path}")
            logger.info(f"Directory contents: {os.listdir(self.local_model_path)}")

            use_8bit = os.getenv("USE_8BIT", "").strip().lower() == "true"
            
            if use_8bit:
                logger.info("Using 8-bit quantization")
                self.tokenizer, self.model = load_model_8bit(model_id, **kwargs)
            else:
                logger.info("Using fp16/bf16 precision")
                self.tokenizer, self.model = load_model_fp16(model_id, **kwargs)

            logger.info(f"Model loaded and distributed. Device map: {self.model.hf_device_map}")

            # Set up generation config
            self.generation_config = self.model.generation_config

            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # Optional: Add optimizations
            sfast_enabled = os.getenv("SFAST", "").strip().lower() == "true"
            if sfast_enabled:
                logger.info(
                    "LLMGeneratePipeline will be dynamically compiled with stable-fast for %s",
                    model_id,
                )
                from app.pipelines.optim.sfast import compile_model
                self.model = compile_model(self.model)

    async def __call__(self, prompt: str, history: Optional[List[tuple]] = None, system_msg: Optional[str] = None, **kwargs) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        conversation = []
        if system_msg:
            conversation.append({"role": "system", "content": system_msg})
        if history:
           conversation.extend(history)
        conversation.append({"role": "user", "content": prompt})

        max_new_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)

        total_tokens = 0
        
        if os.getenv("USE_EXL2", "") != "":
            from exllamav2.generator import ExLlamaV2DynamicJobAsync, ExLlamaV2Sampler
            settings = ExLlamaV2Sampler.Settings(temperature=temperature)
            model_prompt = []
            for msg in conversation:
                model_prompt.append(f"{msg['role']}: {msg['content']}")
            input_ids = self.tokenizer.encode(model_prompt, encode_special_tokens = True)
            job = ExLlamaV2DynamicJobAsync(
                generator = self.generator,
                input_ids = input_ids,
                max_new_tokens = max_new_tokens,
                stop_conditions = self.get_stop_conditions("llama3", self.tokenizer),
                gen_settings = settings,
            )
            
            logger.info("job starting...")
            start = time.time()
            async for result in job:
                text = result.get("text", "")
                if text != "":
                    yield text
                    await asyncio.sleep(0)  # Allow other tasks to run
                if result.get("eos", False):
                    total_tokens = result.get("new_tokens", 0)
            # We'll only collect text here, but the result could contain other updates
            input_length = input_ids.size(1)
            yield {"tokens_used": input_length + total_tokens}
        else:
            input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to(self.model.device)
            attention_mask = torch.ones_like(input_ids)

            streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

            generate_kwargs = self.generation_config.to_dict()
            generate_kwargs.update({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id,
            })

            thread = Thread(target=self.model_generate_wrapper, kwargs=generate_kwargs)
            thread.start()

            try:
                for text in streamer:
                    total_tokens += 1
                    yield text
                    await asyncio.sleep(0)  # Allow other tasks to run
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                raise

            input_length = input_ids.size(1)
            yield {"tokens_used": input_length + total_tokens}

    def model_generate_wrapper(self, **kwargs):
        try:
            logger.debug("Entering model.generate")
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                self.model.generate(**kwargs)
            logger.debug("Exiting model.generate")
        except Exception as e:
            logger.error(f"Error in model.generate: {str(e)}", exc_info=True)
            raise
    
    def get_stop_conditions(self, prompt_format, tokenizer):
        if prompt_format == "llama":
            return [tokenizer.eos_token_id]
        elif prompt_format == "llama3":
            return [tokenizer.single_id("<|eot_id|>")]
        elif prompt_format == "granite":
            return [tokenizer.eos_token_id, "\n\nQuestion:"]
        elif prompt_format == "gemma":
            return [tokenizer.eos_token_id, "<end_of_turn>"]
        
    def __str__(self):
        return f"LLMGeneratePipeline(model_id={self.model_id})"