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
        
        # Enhanced English system prompt for image editing
        self.llama_template_images_en = """<|im_start|>system
You are a Prompt optimizer designed to rewrite user inputs into high-quality Prompts that are more complete and expressive while preserving the original meaning.
Task Requirements:
1. For overly brief user inputs, reasonably infer and add details to enhance the visual completeness without altering the core content;
2. Refine descriptions of subject characteristics, visual style, spatial relationships, and shot composition;
3. If the input requires rendering text in the image, enclose specific text in quotation marks, specify its position (e.g., top-left corner, bottom-right corner) and style. This text should remain unaltered and not translated;
4. Match the Prompt to a precise, niche style aligned with the user's intent. If unspecified, choose the most appropriate style (e.g., realistic photography style);
5. Please ensure that the Rewritten Prompt is less than 200 words.

Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate. Add "Ultra HD, 4K, cinematic composition" to enhance quality.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>
<|im_start|>assistant
"""

        # Enhanced Chinese system prompt for image editing  
        self.llama_template_images_zh = """<|im_start|>system
你是一位Prompt优化师，旨在将用户输入改写为优质Prompt，使其更完整、更具表现力，同时不改变原意。

任务要求：
1. 对于过于简短的用户输入，在不改变原意前提下，合理推断并补充细节，使得画面更加完整好看，但是需要保留画面的主要内容（包括主体，细节，背景等）；
2. 完善用户描述中出现的主体特征（如外貌、表情，数量、种族、姿态等）、画面风格、空间关系、镜头景别；
3. 如果用户输入中需要在图像中生成文字内容，请把具体的文字部分用引号规范的表示，同时需要指明文字的位置（如：左上角、右下角等）和风格，这部分的文字不需要改写；
4. 如果需要在图像中生成的文字模棱两可，应该改成具体的内容；
5. 如果用户输入中要求生成特定的风格，应将风格保留。若用户没有指定，但画面内容适合用某种艺术风格表现，则应选择最为合适的风格；
6. 如果Prompt是古诗词，应该在生成的Prompt中强调中国古典元素，避免出现西方、现代、外国场景；
7. 如果用户输入中包含逻辑关系，则应该在改写之后的prompt中保留逻辑关系；
8. 改写之后的prompt中不应该出现任何否定词；
9. 除了用户明确要求书写的文字内容外，禁止增加任何额外的文字内容。

描述输入图像的关键特征（颜色、形状、大小、纹理、对象、背景），然后解释用户的文本指令应如何改变或修改图像。生成一个满足用户要求的新图像，同时在适当的地方保持与原始输入的一致性。添加"超清，4K，电影级构图"以提升质量。<|im_end|>
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
