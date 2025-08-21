import node_helpers
import comfy.utils
import math
from .qwen_image import QwenImageTokenizer

class TextEncodeQwenImageEditEnhanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "language": (["English", "Chinese"], {"default": "Chinese"}),
            },
            "optional": {"vae": ("VAE", ),
                         "image": ("IMAGE", ),}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, language="Chinese", vae=None, image=None):
        # Create enhanced tokenizer with your improved templates
        tokenizer = QwenImageTokenizer()
        
        # Set template based on language selection
        if language == "Chinese":
            tokenizer.llama_template_images = tokenizer.llama_template_images_zh
        else:
            tokenizer.llama_template_images = tokenizer.llama_template_images_en
        
        # Replace the clip's tokenizer with our enhanced version
        original_tokenizer = clip.tokenizer
        clip.tokenizer = tokenizer
        
        ref_latent = None
        if image is None:
            images = []
        else:
            samples = image.movedim(-1, 1)
            total = int(1024 * 1024)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)
            images = [image[:, :, :, :3]]
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
        
        # Restore original tokenizer
        clip.tokenizer = original_tokenizer
        
        return (conditioning, )


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEditEnhanced": TextEncodeQwenImageEditEnhanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEditEnhanced": "Text Encode Qwen Image Edit (Enhanced)",
}
