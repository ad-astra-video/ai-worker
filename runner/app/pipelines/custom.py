import logging
import os, sys

logger = logging.getLogger(__name__)

custom_pipeline_path = os.environ["CUSTOM_PIPELINE_PATH"]
if os.path.exists(custom_pipeline_path):
    sys.path.append(custom_pipeline_path)

from custom_pipeline import *

class CustomPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {"cache_dir": get_model_dir()+"/custom"}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        if not os.path.exists(folder_path):
            raise ValueError("custom model folder does not exists, exiting")
        
        if load_custom_model_path():
            try:
                import custom_pipeline
                self.ldm = custom_pipeline.load_model(model_id, **kwargs)
            except:
                raise ValueError("custom model could not be loaded, no load_model in custom_pipeline module")
        else:
            raise ValueError("custom model could not be loaded, verify path exists and load_model function is in scope of path")
        
        #not sure how the safetychecker can be used on custom models
        #safety_checker_device = os.getenv("SAFETY_CHECKER_DEVICE", "cuda").lower()
        #self._safety_checker = SafetyChecker(device=safety_checker_device)

    def __call__(
        self, prompt: str, image: PIL.Image, **kwargs
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:
        seed = kwargs.pop("seed", None)
        num_inference_steps = kwargs.get("num_inference_steps", None)
        safety_check = kwargs.pop("safety_check", True)

        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(get_torch_device()).manual_seed(
                    seed
                )
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(get_torch_device()).manual_seed(s) for s in seed
                ]

        if num_inference_steps is None or num_inference_steps < 1:
            del kwargs["num_inference_steps"]

        if (
            self.model_id == "stabilityai/sdxl-turbo"
            or self.model_id == "stabilityai/sd-turbo"
        ):
            # SD turbo models were trained without guidance_scale so
            # it should be set to 0
            kwargs["guidance_scale"] = 0.0

            # Ensure num_inference_steps * strength >= 1 for minimum pipeline
            # execution steps.
            if "num_inference_steps" in kwargs:
                kwargs["strength"] = max(
                    1.0 / kwargs.get("num_inference_steps", 1),
                    kwargs.get("strength", 0.5),
                )
        elif ModelName.SDXL_LIGHTNING.value in self.model_id:
            # SDXL-Lightning models should have guidance_scale = 0 and use
            # the correct number of inference steps for the unet checkpoint loaded
            kwargs["guidance_scale"] = 0.0

            if "2step" in self.model_id:
                kwargs["num_inference_steps"] = 2
            elif "4step" in self.model_id:
                kwargs["num_inference_steps"] = 4
            elif "8step" in self.model_id:
                kwargs["num_inference_steps"] = 8
            else:
                # Default to 2step
                kwargs["num_inference_steps"] = 2

        output = self.ldm(prompt, image=image, **kwargs)

        if safety_check:
            _, has_nsfw_concept = self._safety_checker.check_nsfw_images(output.images)
        else:
            has_nsfw_concept = [None] * len(output.images)

        return output.images, has_nsfw_concept

    def __str__(self) -> str:
        return f"ImageToImagePipeline model_id={self.model_id}"
