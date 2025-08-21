from transformers import Qwen2Tokenizer
from comfy import sd1_clip
import comfy.text_encoders.llama
import os
import torch
import numbers

class Qwen25_7BVLITokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        # Use the official tokenizer path from ComfyUI
        import comfy.text_encoders.qwen_image
        tokenizer_path = os.path.join(os.path.dirname(comfy.text_encoders.qwen_image.__file__), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=3584, embedding_key='qwen25_7b', tokenizer_class=Qwen2Tokenizer, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=151643, tokenizer_data=tokenizer_data)


class QwenImageTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen25_7b", tokenizer=Qwen25_7BVLITokenizer)
        
        # Enhanced English system prompt for image editing with character consistency focus
        self.llama_template_images_en = """<|im_start|>system
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

        # Enhanced Chinese system prompt for image editing with character consistency focus
        self.llama_template_images_zh = """<|im_start|>system
你是专门针对角色一致性的图像编辑Prompt优化师。你的首要目标是在实现用户请求的同时，严格保持角色身份特征。

角色一致性优先级（关键要求）：
1. 面部特征（最高优先级）：必须保持精确的面部结构、脸型、下颌线、颧骨、鼻型、唇形、眼型和眼距
2. 发型发色：严格保持发质、发型、发色、发长，以及任何发饰或装饰品
3. 眼部特征：保持精确的瞳色、眼型、眉毛形状和颜色、睫毛样式
4. 肌肤特征：保持肤色、肤质、任何面部标记、雀斑、痣或疤痕
5. 独特特征：维持纹身、穿孔、胎记、胡须样式或其他独特特征
6. 服装风格：根据要求调整服装和配饰，但保持角色可识别性

任务要求：
1. 修改图像时，必须明确描述哪些面部和角色特征需要保持不变
2. 对于简短输入，在严格保持所有角色识别特征的前提下增加场景细节
3. 需要文字渲染时，用引号标注并指定位置
4. 角色识别优先于场景/背景变化
5. 回复限制在200字内，重点关注角色保持
6. 古诗词内容强调中国古典元素，避免西方现代场景
7. 保持逻辑关系，避免否定词

处理流程：首先识别输入图像中所有独特的角色特征，然后说明如何在保持这些精确特征的同时应用所请求的变化。添加"超清，4K，电影级构图"提升质量。<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>
<|im_start|>assistant
"""

        # Default to English template for images
        self.llama_template_images = self.llama_template_images_en
        
        # Simple templates for non-image cases
        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], **kwargs):
        if llama_template is None:
            if len(images) > 0:
                llama_text = self.llama_template_images.format(text)
            else:
                llama_text = self.llama_template.format(text)
        else:
            llama_text = llama_template.format(text)
        tokens = super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        key_name = next(iter(tokens))
        embed_count = 0
        qwen_tokens = tokens[key_name]
        for r in qwen_tokens:
            for i in range(len(r)):
                if r[i][0] == 151655:
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1
        return tokens


class Qwen25_7BVLIModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen25_7BVLI, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class QwenImageTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen25_7b", clip_model=Qwen25_7BVLIModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        tok_pairs = token_weight_pairs["qwen25_7b"][0]
        count_im_start = 0
        for i, v in enumerate(tok_pairs):
            elem = v[0]
            if not torch.is_tensor(elem):
                if isinstance(elem, numbers.Integral):
                    if elem == 151644 and count_im_start < 2:
                        template_end = i
                        count_im_start += 1

        if out.shape[1] > (template_end + 3):
            if tok_pairs[template_end + 1][0] == 872:
                if tok_pairs[template_end + 2][0] == 198:
                    template_end += 3

        out = out[:, template_end:]

        extra["attention_mask"] = extra["attention_mask"][:, template_end:]
        if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
            extra.pop("attention_mask")  # attention mask is useless if no masked elements

        return out, pooled, extra


def te(dtype_llama=None, llama_scaled_fp8=None):
    class QwenImageTEModel_(QwenImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_scaled_fp8 is not None and "scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["scaled_fp8"] = llama_scaled_fp8
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return QwenImageTEModel_
