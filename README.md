# Qwen-image/qwen-image-edit-自定义系统提示词（文本编码器）


## 最新动作
- 将系统提示词更改为可自定义的方式以增强灵活性
- 增加了对于qwen-image文生图场景的支持

<img width="2172" height="1614" alt="image" src="https://github.com/user-attachments/assets/221e0e7a-4c62-42f6-92e3-6a76beb00cfe" />
<img width="1161" height="749" alt="image" src="https://github.com/user-attachments/assets/79855c98-9149-4b55-bf80-054fa8be51f5" />



## 问题背景

ComfyUI 官方的 `TextEncodeQwenImageEdit` 节点简化了提示词工程，导致角色一致性变差。

## 解决方案

本自定义节点完全替代官方节点，支持系统提示词自定义
如果你不会写，看这里：https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils.py

## 使用方法

1. 重启 ComfyUI
2. 搜索节点：`Text Encode Qwen Image Edit (Enhanced)`
3. 替换原有的 `TextEncodeQwenImageEdit` 节点


## 核心改进

- 支持系统提示词的自定义

独立维护，不受官方简化版本限制。
