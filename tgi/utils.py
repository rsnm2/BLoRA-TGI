import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel, Field
from queue import Queue
from enum import Enum
from typing import List, Optional, Tuple
from logits_process import (RepetitionPenaltyLogitsProcessor, LogitsWarpers, softmax)

# TODO: convert back to torch implementation
# TODO: sample for b > 1 with vectorized code
class Greedy:
    def __call__(self, logits: np.ndarray):
        # assert b=1 for now
        # shape == (batch, vocabulary_size)
        assert(logits.shape[0] == 1)
        assert(len(logits.shape) == 2)      
        
        return np.argmax(logits[0,:])

# TODO: convert back to torch implementation
# TODO: sample for b > 1 with vectorized code
# https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a
class Sampling:
    def __init__(self, seed:int=42):
        self.generator = np.random.default_rng(seed=seed)
    
    def __call__(self, logits:np.ndarray):
        # assert b=1 for now
        # shape == (batch, vocabulary_size)
        assert(logits.shape[0] == 1)
        assert(len(logits.shape) == 2)
        
        probs = softmax(logits, axis=1)
        return self.generator.choice(probs.shape[1], p=probs[0,:])

# TODO: convert back to torch implementation
class NextTokenChooser:
    def __init__(
        self,
        repetition_penalty: Optional[float] = 1.0,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample:bool = False,
        seed: int = 42,
    ):
        self.repetition_processor = (
            RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
            if repetition_penalty and repetition_penalty != 1.0 else None
        )
        
        has_warpers = (
            (temperature is not None and temperature != 1.0)
            or (top_k is not None and top_k != 0)
            or (top_p is not None and top_p < 1.0)
        )

        if has_warpers:
            self.warpers = LogitsWarpers(temperature=temperature, top_k=top_k, top_p=top_p)
        else:
            self.warpers = None

        self.choice = Sampling(seed=seed) if do_sample or has_warpers else Greedy()

    def __call__(self, input_ids: np.ndarray, scores:np.ndarray) -> int:
        if self.repetition_processor is not None:
            scores = self.repetition_processor(input_ids=input_ids, scores=scores)

        if self.warpers is not None:
            scores = self.warpers(scores=scores)
        
        return self.choice(scores)
        
class FinishReason(Enum):
    FINISH_REASON_LENGTH = 1
    FINISH_REASON_EOS_TOKEN = 2

class StoppingCriteria:
    def __init__(
        self,
        eos_token_id: int,    
        max_new_tokens: int = 20,
        ignore_eos_token: bool = False,
    ):
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.current_tokens = 0
        self.ignore_eos_token = ignore_eos_token

    def __str__(self):
        return f"[StoppingCriteria.max_new_tokens: {self.max_new_tokens}]"

    def __repr__(self):
        return f"[StoppingCriteria.max_new_tokens: {self.max_new_tokens}]"
    
    def __call__(self, token_id: int) -> Tuple[bool, Optional[str]]:
        self.current_tokens += 1
        if self.current_tokens >= self.max_new_tokens:
            return True, FinishReason.FINISH_REASON_LENGTH

        if not self.ignore_eos_token and token_id == self.eos_token_id:
            return True, FinishReason.FINISH_REASON_EOS_TOKEN

        return False, None
    
    def remaining_decode_tokens(self):
        return self.max_new_tokens - self.current_tokens - 1

# TODO: convert to pydantic
class GenerateParameters(BaseModel):
    max_new_tokens: int = Field(default=20)
    # repetition_penalty: float = Field(default=1)
    # do_sample: bool = Field(default=False)
    # temperature: float = Field(default=1.0)
    # top_k: Optional[int] = Field(default=None)
    # top_p: Optional[float] = Field(default=None)
    # seed: int = Field(default=42)

# TODO: convert to pydantic
class GenerateRequestInputs(BaseModel):
    inputs: str
    lora_id: str
    generate_parameters: GenerateParameters

# TODO: convert to pydantic
class GenerateRequestOutputs(BaseModel):
    response_text: str = Field(default="")
    finish_reason: Optional[FinishReason] = Field(default=None)

@dataclass
class Request:
    id: int
    inputs: str
    lora_id: str
    generate_parameters: GenerateParameters

@dataclass
class Batch:
    id: int
    requests: List[Request]

@dataclass
class CachedBatch:
    batch_id: int
    request_ids: List[int]

    def __len__(self):
        return len(self.request_ids)

@dataclass
class Generation:
    request_id: int
    token_id: Optional[int]
    stopped: bool
    finish_reason: FinishReason = None

@dataclass
class GenerateRequest:
    inputs: str
    lora_id: str
    generate_parameters: GenerateParameters
    response_stream: "Queue[Generation]"

    @classmethod
    def from_gr_inputs(cls, gr_inputs: GenerateRequestInputs):
        return cls(
            inputs=gr_inputs.inputs,
            lora_id=gr_inputs.lora_id,
            generate_parameters=gr_inputs.generate_parameters,
            response_stream=Queue()
        )