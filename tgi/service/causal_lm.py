import torch, inspect
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, LlamaTokenizer

from utils import Request, Batch, CachedBatch, Generation, StoppingCriteria, NextTokenChooser
from service.blora_utils import load_loras

MAX_GENERATED_TOKENS = 512
MAX_TRUNCATION = 256

@dataclass
class BLoraCausalLMBatch:    
    batch_id: int
    requests: List[Request]
    requests_idx_mapping: Dict[int,int]
    lora_ids: List[str]

    # decoder values
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: Optional[List[Tuple]]

    # generation / state helpers
    input_lengths: List[int]
    max_input_length: int
    padding_right_offset: int
    
    @classmethod
    def from_batch(
        cls,
        batch: Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "BLoraCausalLMBatch":
        
        # parse batch
        requests_idx_mapping = {}
        lora_ids = []
        inputs = []
        max_decode_tokens = 0
        padding_right_offset = 0
        for idx, r in enumerate(batch.requests):
            requests_idx_mapping[r.id] = idx
            lora_ids.append(r.lora_id)
            inputs.append(r.inputs)
            
            # TODO: update to use stopping criteria instead
            padding_right_offset = max(padding_right_offset, MAX_GENERATED_TOKENS)
            max_decode_tokens += MAX_GENERATED_TOKENS

        # run tokenizer
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            max_length=MAX_TRUNCATION
        ).to(device)
        
        # attention ids
        input_ids = tokenized_inputs["input_ids"]
        
        # create fully allocated attention mask (shaped [b, max_total_tokens]), 
        # copy in tokenized section
        input_lengths = tokenized_inputs["attention_mask"].sum(1)
        max_input_length = input_lengths.max()
        attention_mask = input_ids.new_zeros(len(inputs), max_input_length + padding_right_offset)
        attention_mask[:, :max_input_length] = tokenized_inputs["attention_mask"]
        
        # setup position ids
        position_ids = tokenized_inputs["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(tokenized_inputs["attention_mask"] == 0, 1)

        return cls(
            batch_id=batch.id,
            requests=batch.requests,
            requests_idx_mapping=requests_idx_mapping,
            lora_ids=lora_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            input_lengths=input_lengths.tolist(),
            max_input_length=max_input_length.item(),
            padding_right_offset=padding_right_offset,
        )

    def to_cached_batch(self) -> CachedBatch:
        return CachedBatch(
            batch_id = self.batch_id,
            request_ids=[r.id for r in self.requests],
        )

    # length of the batch
    def __len__(self):
        return len(self.requests)

#     # pass list of request ids, returns batch with only those request ids
#     def filter(self, request_ids: List[int]) -> Optional["DeepSparseCausalLMBatch"]:
#         assert(len(request_ids) > 0)

#         requests_idx_mapping    = {}
#         requests                = []
#         input_ids_list          = []
#         past_key_values_list    = []
#         stopping_criteria_list  = []
#         next_token_chooser_list = []

#         # loop through requests, keep ones that should remain
#         for new_idx, request_id in enumerate(request_ids):
#             assert request_id in self.requests_idx_mapping.keys(), "all request ids must be in the batch"
            
#             requests_idx_mapping[request_id] = new_idx
            
#             old_idx = self.requests_idx_mapping[request_id]
#             requests.append(self.requests[old_idx])
#             input_ids_list.append(self.input_ids_list[old_idx])
#             past_key_values_list.append(self.past_key_values_list[old_idx])
#             stopping_criteria_list.append(self.stopping_criteria_list[old_idx])
#             next_token_chooser_list.append(self.next_token_chooser_list[old_idx])

#         # update batch state
#         self.requests = requests
#         self.requests_idx_mapping = requests_idx_mapping 
#         self.input_ids_list = input_ids_list
#         self.past_key_values_list = past_key_values_list
#         self.stopping_criteria_list = stopping_criteria_list
#         self.next_token_chooser_list = next_token_chooser_list

#         return self

#     # combine two batches into one
#     @classmethod
#     def concatenate(cls, batches: List["DeepSparseCausalLMBatch"]) -> "DeepSparseCausalLMBatch":
#         assert len(batches) > 1, "must have more than 1 batch to concatenate"

#         requests_idx_mapping    = {}
#         requests                = []
#         input_ids_list          = []
#         past_key_values_list    = []
#         stopping_criteria_list  = []
#         next_token_chooser_list = []

#         start_index = 0
#         for i, batch in enumerate(batches):
#             assert batch.past_key_values_list is not None, "only concatenate prefilled batches"
            
#             # concatenate request, input_ids, and past_key_values lists
#             requests.extend(batch.requests)
#             input_ids_list.extend(batch.input_ids_list)
#             past_key_values_list.extend(batch.past_key_values_list)
#             stopping_criteria_list.extend(batch.stopping_criteria_list)
#             next_token_chooser_list.extend(batch.next_token_chooser_list)

#             # merge the request_id to index mapping
#             if i == 0:
#                 requests_idx_mapping = batch.requests_idx_mapping
#             else:
#                 for k, v in batch.requests_idx_mapping.items():
#                     requests_idx_mapping[k] = v + start_index
            
#             start_index += len(batch)

#         return cls(
#             batch_id=batches[0].batch_id,
#             requests=requests,
#             requests_idx_mapping=requests_idx_mapping,
#             input_ids_list=input_ids_list,
#             past_key_values_list=past_key_values_list,
#             stopping_criteria_list=stopping_criteria_list,
#             next_token_chooser_list=next_token_chooser_list
#         )

class BLoraCausalLM:
    def __init__(
        self,
        base_model_id: str,
        lora_ids: List[str],
        has_position_ids: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        self.active_batch_id = None
        
        # load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto", 
            torch_dtype=dtype
        )
        print("Done!\n")

        # load loras
        print("Loading LORAs...")
        self.model, self.lora_map = load_loras(self.model, lora_ids)
        print("Done!\n")

        # check if model takes position ids
        self.has_position_ids = has_position_ids

        # setup tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_id)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None:
            assert self.tokenizer.eos_token_id is not None
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def set_batch_lora_ids(
        self,
        batch_id: int,
        batch_lora_ids: List[str],
    ):  
        # if current batch is not already active, set lora ids
        if self.active_batch_id is None or self.active_batch_id != batch_id:
            self.active_batch_id = batch_id

            inp_loras = [self.lora_map[lora_id] for lora_id in batch_lora_ids]
            # TODO - figure out if this take a long time and make it faster
            for _, module in self.model.named_modules():
                module.batch_lora_ids = inp_loras

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
        }

        if self.has_position_ids:
            kwargs["position_ids"] = position_ids

        outputs = self.model.forward(**kwargs)
        return outputs.logits, outputs.past_key_values

    def greedy(
        self,
        logits
    ):
        assert(len(logits.shape) == 2)
        return logits[-1,:].argmax()        
        
    def eval_stopping_criteria(
        self,
        token_id,
        input_length
    ):
        return (
            # model tells us to stop
            token_id == self.tokenizer.eos_token_id or 
            # we have reached the max tokens to generate
            input_length > MAX_GENERATED_TOKENS
        )

    def generate_token(
        self,
        batch: BLoraCausalLMBatch,
    ) -> (List[Generation], Optional[BLoraCausalLMBatch]):
        
        # set loras to match the current batch
        self.set_batch_lora_ids(batch.batch_id, batch.lora_ids)

        # run forward pass
        attention_mask = batch.attention_mask[:, : -batch.padding_right_offset]
        logits, past = self.forward(
            batch.input_ids,
            attention_mask,
            batch.position_ids,
            batch.past_key_values,
        )

        generations: List[Generation] = []
        all_stopped = True

        # for each member of the batch:
        #   a) sample, b) check stopping criteria, c) create generation, d) update batch
        iterator = zip(
            batch.requests, 
            batch.input_lengths, 
            logits,
        )
        for i, (
            request, 
            input_length, 
            logits,
        ) in enumerate(iterator):
            
            # (a) sample
            next_token_id = self.greedy(logits)
            next_token = self.tokenizer.decode(next_token_id)
            new_input_length = input_length + 1

            # (b) check stopping
            stop = self.eval_stopping_criteria(
                token_id=next_token_id, 
                input_length=new_input_length
            )
            if not stop:
                all_stopped = False
  
            # c) make generation
            generations.append(Generation(
                request_id=request.id,
                token=next_token,
                token_id=next_token_id,
                stopped=stop,
                # finish_reason=finish_reason
            ))

            # (d) update batch
            batch.input_ids[i, 0] = next_token_id
            batch.input_lengths[i] = new_input_length
            batch.max_input_length = max(batch.max_input_length, new_input_length)

        # we finished all generations, there is no next batch
        if all_stopped:
            return generations, None
        
        # otherwise, update the batch inputs
        batch.input_ids = batch.input_ids[:, :1]                    # slice off unnessary ids from prefill
        batch.attention_mask[:, -batch.padding_right_offset] = 1    # sequence is now 1 longer (one less pad token)
        batch.padding_right_offset -= 1
        batch.position_ids = batch.position_ids[:, -1:] + 1         # one more item in the sequence
        batch.past_key_values = past                                # torch module handled updating this

        return generations, batch