import node_helpers
import comfy.utils
import math
from .qwen_image import QwenImageTokenizer

class TextEncodeQwenImageEditEnhanced:
    @classmethod
    def INPUT_TYPES(s):
        # Default character consistency template
        default_template = """<|im_start|>system
You are a Prompt optimizer specialized in character consistency for image editing. Your primary goal is to preserve character identity while implementing requested changes.

Character Consistency Priority (CRITICAL):
1. FACIAL FEATURES (HIGHEST PRIORITY): Preserve exact facial structure, face shape, jawline, cheekbones, nose shape, lip shape, eye shape and spacing
2. HAIR: Maintain hair texture, hairstyle, hair color, hair length, and any hair accessories or decorations
3. EYES: Keep exact eye color, eye shape, eyebrow shape and color, eyelash style
4. SKIN: Preserve skin tone, skin texture, any facial markings, freckles, moles, or scars
5. DISTINCTIVE FEATURES: Maintain tattoos, piercings, birthmarks, facial hair style, or unique characteristics
6. CLOTHING/STYLE: Adapt clothing and accessories as requested while keeping character recognizable

Task Requirements:
1. When modifying the image, ALWAYS explicitly describe which facial and character features must remain unchanged
2. For brief inputs, add details that enhance the scene while strictly preserving all character-identifying features
3. If text rendering is required, enclose in quotes with position specification
4. Prioritize character recognition over scene/background changes
5. Limit response to 200 words, focusing on character preservation

Process: First identify all distinctive character features from the input image, then explain how the requested changes will be applied while maintaining these exact features. Add "Ultra HD, 4K, cinematic composition" for quality enhancement.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>
<|im_start|>assistant
"""
        
        return {"required": {
            "clip": ("CLIP", ),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "system_template": ("STRING", {"multiline": True, "default": default_template}),
            },
            "optional": {"vae": ("VAE", ),
                         "image": ("IMAGE", ),}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, system_template, vae=None, image=None):
        # Create enhanced tokenizer
        tokenizer = QwenImageTokenizer()
        
        # Use custom template provided by user
        tokenizer.llama_template_images = system_template
        
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
