import torch
from typing import Dict, List, Tuple
from service.causal_lm import BLoraCausalLM, BLoraCausalLMBatch
from utils import Generation, CachedBatch, Batch

class BatchCache:
    def __init__(self):
        self.cache: Dict[int, BLoraCausalLMBatch] = {}

    def pop(self, batch_id: int) -> BLoraCausalLMBatch:
        batch = self.cache.pop(batch_id, None)
        if batch is None:
            raise ValueError(f"Batch ID {batch_id} not found in cache.")
        return batch

    def set(self, entry: BLoraCausalLMBatch):
        if entry is not None:
            self.cache[entry.batch_id] = entry

    def delete(self, batch_id: int):
        batch = self.pop(batch_id)
        if batch is not None:
            del batch

    def clear(self):
        keys = list(self.cache.keys())
        for k in keys:
            self.delete(k)

    def __len__(self):
        return len(self.cache.keys())

class TextGenerationService:
    def __init__(
        self, 
        base_model_id: str,
        lora_ids: str
    ):
        self.model = BLoraCausalLM(
            base_model_id=base_model_id,
            lora_ids=lora_ids
        )
        self.cache = BatchCache()

    def ClearCache(self):
        self.cache.clear()

    def FilterBatch(self, batch_id: int, request_ids: List[int]) -> CachedBatch:
        clm_batch = self.cache.pop(batch_id)
        filtered_clm_batch = clm_batch.filter(request_ids)
        self.cache.set(filtered_clm_batch)

        return filtered_clm_batch.to_cached_batch()

    def Prefill(self, batch: Batch) -> Tuple[List[Generation], CachedBatch]:
        clm_batch = BLoraCausalLMBatch.from_batch(
            batch=batch,
            tokenizer=self.model.tokenizer,
            device="cuda" if torch.cuda.is_available else "cpu"
        )

        generations, next_clm_batch = self.model.generate_token(clm_batch)
        self.cache.set(next_clm_batch)

        return generations, (next_clm_batch.to_cached_batch() if next_clm_batch else None)

    def Decode(self, batches: List[CachedBatch]) -> Tuple[List[Generation], CachedBatch]:
        if len(batches) <= 0:
            raise ValueError("Must pass at least one batch to service.Decode")

        clm_batches = []
        for cached_batch in batches:
            clm_batches.append(self.cache.pop(cached_batch.batch_id))

        if len(clm_batches) > 1:
            clm_batch = BLoraCausalLMBatch.concatenate(clm_batches)
        else:
            clm_batch = clm_batches[0]

        generations, next_clm_batch = self.model.generate_token(clm_batch)
        self.cache.set(next_clm_batch)

        return generations, (next_clm_batch.to_cached_batch() if next_clm_batch else None)