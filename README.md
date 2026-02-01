# LITE
Code for Muon-LITE on pre-training LLMs


IMPORTANT: To reproduce Qwen2-MoE experiments, strict version control of `transformers` is required. Versions too low lack `DataCollatorWithFlattening`, while utilizing the official implementation in newer versions  triggers automatic fusion of the up/gate layers (and some other unknown errors). We recommend using appropriate versions such as 4.51.0.
