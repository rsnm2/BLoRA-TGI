from transformers import AutoModelForCausalLM, AutoTokenizer
from blora_utils import load_loras
import torch
from typing import Optional, List, Dict

class TGIBLoraModel:
    def __init__(
        self,
        base_model_id: str,
        lora_ids: List[str],
    ):
        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )

        # load loras
        self.model, self.lora_map = load_loras(self.model, lora_ids)

    # forward pass
    def __call__(
        self,
        input_ids,
        attention_mask,
        past_key_values = None,
    ):
        
        # actual forward pass
        logits, past_key_values = self.model.forward(
            input_ids,
            attention_mask,
            past_key_values,
            use_cache=True
        )

        return logits, past_key_values