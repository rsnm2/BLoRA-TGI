import torch

torch.set_default_tensor_type(torch.cuda.HalfTensor)

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, LlamaTokenizer

from utils import Request, Batch, CachedBatch, Generation, StoppingCriteria, NextTokenChooser
from service.blora_utils import load_loras

MAX_TRUNCATION = 256

@dataclass
class BLoraCausalLMBatch:    
    # metadata
    batch_id: int
    requests: List[Request]
    requests_idx_mapping: Dict[int,int]
    lora_ids: List[str]

    # model inputs
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: Optional[List[Tuple]]

    # generation helpers
    stopping_criterias: List[StoppingCriteria]

    # padding metadata
    input_lengths: List[int]
    max_input_length: int
    padding_right_offset: int

    def p(self):
        for key in self.__dict__:
            if key == "past_key_values" and self.past_key_values is None:
                val = "None"
            elif key == "past_key_values" and self.past_key_values is not None:
                val = f"{self.past_key_values[0][0].shape}"
            else:
                val = getattr(self, key)
                if type(val) == torch.Tensor:
                    print(f"{key}.shape: {val.shape}")
                
            print(f"{key}: {val}")

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
        stopping_criterias = []
        padding_right_offset = 0

        for idx, r in enumerate(batch.requests):
            requests_idx_mapping[r.id] = idx
            lora_ids.append(r.lora_id)
            inputs.append(r.inputs)
            
            max_new_tokens = r.generate_parameters.max_new_tokens
            stopping_criterias.append(StoppingCriteria(
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                ignore_eos_token=False,
            ))

            padding_right_offset = max(padding_right_offset, max_new_tokens - 1)

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
            stopping_criterias=stopping_criterias,
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

    # 1) update metatas
    # 2) update model inputs (reshaping / removing padding if needed)
    #       note:   we need to remove padding in cases where the item filtered
    #               was bounding the kv cache sizes / attention mask sizes
    # 3) set new batch state
    def filter(self, request_ids: List[int]) -> Optional["BLoraCausalLMBatch"]:
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            raise ValueError("No change to batch size, nothing to Filter")

        keep_indices = []
        
        # -------------------------------------------------------------------------------- #
        # 1) UPDATE METADATAS ------------------------------------------------------------ #
        # -------------------------------------------------------------------------------- #
        #   a) metata, b) generation, c) padding metadata

        # metadata
        new_requests = []
        new_lora_ids = []
        new_requests_idx_mapping = {}

        # generation helpers
        new_stopping_criterias = []
        
        # padding metadata
        new_input_lengths = []
        new_max_input_length = 0
        new_padding_right_offset = 0

        # loop through requests, keep ones that should remain
        for new_idx, request_id in enumerate(request_ids):
            if request_id not in self.requests_idx_mapping.keys():
                raise ValueError("All request ids passed to filter() must be in the batch")
            
            # get old idx of the this request, update mapping
            old_idx = self.requests_idx_mapping[request_id]
            keep_indices.append(old_idx)

            # a) new metadata
            new_requests.append(self.requests[old_idx])
            new_requests_idx_mapping[request_id] = new_idx
            new_lora_ids.append(self.lora_ids[old_idx])

            # b) new genertation helpers
            new_stopping_criterias.append(self.stopping_criterias[old_idx])
            
            # c) new padding metadata
            new_input_lengths.append(self.input_lengths[old_idx])
            new_max_input_length = max(
                new_max_input_length, 
                new_input_lengths[-1]
            )
            new_padding_right_offset = max(
                new_padding_right_offset, 
                new_stopping_criterias[-1].remaining_decode_tokens()
            )

        # -------------------------------------------------------------------------------- #
        # 2) UPDATE MODEL INPUTS --------------------------------------------------------- #
        # -------------------------------------------------------------------------------- #

        # a) filter input ids
        input_ids = self.input_ids[keep_indices]
        
        # b) filter position ids
        position_ids = self.position_ids[keep_indices]
        
        # c) filter attention mask, handling two cases:
        #       c_1: filter enables us to remove left padding
        #       c_2: filter enables us to remove right padding
    
        # c_1) handles case where the item filtered was the longest input in the batch
        #       --> need to chop off the padding tokens to the left
        #
        #       [1,1,1,1,0,0]   << if this item filtered
        #       [0,0,1,1,0,0]
        #       [0,0,0,1,0,0]
        #
        #       [1,1,0,0]       << updated batch should look like this
        #       [0,1,0,0]

        active_start_idx = -(self.padding_right_offset + new_max_input_length)
        
        # c_2) handles case where the item filtered has the longest potential decode
        #       --> need to chop off the padding token tot the right
        #       
        #       [1,1,0,0,0,0,0]   << if this item filtered (and can generate 5 more tokens)
        #       [1,1,0,0,0,0,0]   << and this item can only generate up to 3 more tokens 
        #
        #       [1,1,0,0,0]       << updated batch should look like this

        active_end_idx = (
            self.attention_mask.shape[1] - self.padding_right_offset + new_padding_right_offset
        )

        self.attention_mask = self.attention_mask[
            keep_indices, 
            active_start_idx:active_end_idx
        ]

        # d) update past_key_values tensors, handling case:
        #       d_1: filter enables us to remove left padding
        
        # d_1) handles case where item filter was the longest input in the batch
        #       --> need to chop off the kvs to the left
        #           
        #       [kv0,kv1,kv2,kv3]   << if this item filtered
        #       [pad,pad,kv0,kv1]
        #       [pad,pad,pad,kv0]
        #
        #       [kv0,kv1]           << updated batch should look like this 
        #       [pad,kv0]
        
        # ensure update happens in place (tuples are immutable), for incremental garbage collection 
        if type(self.past_key_values[0]) == tuple:
            self.past_key_values = [list(layer) for layer in self.past_key_values]

        if len(self.past_key_values[0][0].shape) != 4:
            raise NotImplementedError("Only supporting models with past_key_values shape == 4 - i.e. not BLOOM")
        
        # new kv_length is just the longest remaining input - 1
        past_kv_length = new_max_input_length - 1

        # filter each layer
        for layer in self.past_key_values:
            past_keys, past_values = layer

            layer[0] = past_keys[keep_indices, :, -past_kv_length:, :]
            del past_keys
            layer[1] = past_values[keep_indices, :, -past_kv_length:, :]
            del past_values    

        # -------------------------------------------------------------------------------- #
        # 3) UPDATE BATCH STATE ---------------------------------------------------------- #
        # -------------------------------------------------------------------------------- #

        # metadata
        self.requests = new_requests
        self.requests_idx_mapping = new_requests_idx_mapping
        self.lora_ids = new_lora_ids

        # model inputs
        self.input_ids = input_ids
        self.position_ids = position_ids
        # self.attention_mask (updated in place)
        # self.past_key_values (updated in place)
        
        # generation helpers
        self.stopping_criterias = new_stopping_criterias

        # padding metadata
        self.input_lengths = new_input_lengths
        self.max_input_length = new_max_input_length
        self.padding_right_offset = new_padding_right_offset

        return self

    @classmethod
    def concatenate(cls, batches: List["BLoraCausalLMBatch"]) -> "BLoraCausalLMBatch":
        if len(batches) <= 1:
            raise ValueError(f"len(batches) = {len(batches): Must pass more than 1 batch to concatenate}")

        # compute new padding metadata
        total_batch_size = 0
        max_input_length = 0
        padding_right_offset = 0

        for batch in batches:
            total_batch_size += len(batch)
            max_input_length = max(max_input_length, batch.max_input_length)
            padding_right_offset = max(padding_right_offset, batch.padding_right_offset)

        # metadata
        requests = []
        requests_idx_mapping = {}
        lora_ids = []

        # model inputs
        input_ids = None
        attention_mask = None
        position_ids = None
        past_key_values = []

        # generation helpers
        stopping_criterias = []
        
        # padding metadata
        input_lengths = []
        
        start_index = 0
        for i, batch in enumerate(batches):
            end_index = start_index + len(batch)

            # combine all lists
            requests.extend(batch.requests)
            lora_ids.extend(batch.lora_ids)
            stopping_criterias.extend(batch.stopping_criterias)
            input_lengths.extend(batch.input_lengths)        
            
            # update {request_id: list_idx} mapping
            for r_id, l_idx in batch.requests_idx_mapping.items():
                requests_idx_mapping[r_id] = l_idx + start_index
            
            # update model inputs
            if batch.past_key_values is None:
                raise ValueError("Only concatenate prefilled batches")
            
            # a) input_ids 
            # b) position_ids
            #       always of shape [batch_size, 1], no padding needed
            if input_ids is None or position_ids is None:
                assert input_ids is None and position_ids is None
                input_ids = batch.input_ids.new_empty((total_batch_size, 1))
                position_ids = batch.position_ids.new_empty((total_batch_size, 1))

            input_ids[start_index:end_index] = batch.input_ids
            position_ids[start_index:end_index] = batch.position_ids
            
            # c) attention_mask 
            #       padded to [b, max_possible_tokens] == [b, max_input_len, padding_right_offset]
            #
            #       handle cases like the following concat(b1,b2)
            #           b1=
            #           [0,0,1,1,0]
            #           [1,1,1,1,0]
            #
            #           b2=
            #           [1,1,0,0]
            #           [0,1,0,0]
            #
            #           new_attn_mask=
            #           [0,0,1,1,0,0] << inserted old_idx 0-5 to new_idx 0-5
            #           [1,1,1,1,0,0] << inserted old_idx 0-5 to new_idx 0-5
            #           [0,0,1,1,0,0] << inserted old_idx 0-4 to new_idx 2-6
            #           [0,0,0,1,0,0] << inserted old_idx 0-4 to new_idx 2-6 --> new_idx left offset == max_input_len - b2.max_input_lenght
            
            if attention_mask is None:
                attention_mask = batch.attention_mask.new_zeros((total_batch_size, max_input_length + padding_right_offset),)
            
            left_offset = max_input_length - batch.max_input_length
            assert batch.attention_mask.shape[1] - batch.max_input_length - batch.padding_right_offset == 0
            assert batch.attention_mask.shape[1] - batch.padding_right_offset == attention_mask.shape[1] - padding_right_offset - left_offset
            attention_mask[
                start_index:end_index,
                left_offset:-padding_right_offset
            ] = batch.attention_mask[
                :, 
                :-batch.padding_right_offset
            ]
            
            # d) past_key_values
            #       concat logic outside for loop (below)
            #       ensure update happens in place (tuples are immutable), for incremental garbage collection 
            if type(batch.past_key_values[0]) == tuple:
                batch.past_key_values = [list(layer) for layer in batch.past_key_values]
                if len(batch.past_key_values[0][0].shape) != 4:
                    raise NotImplementedError("Only supporting models with past_key_values shape == 4 - i.e. not BLOOM")
            else:
                # TODO: I think this might be okay
                raise ValueError("Currently only supporting tuple types for past_key_values")

            start_index = end_index    
        
        # d) past_key_values
        first_past_key_values = batches[0].past_key_values
        _, num_heads, _, head_dim = first_past_key_values[0][1].shape        
        padded_past_shape = (total_batch_size, num_heads, max_input_length - 1, head_dim)

        # concatenate past kvs layer by layer to allow incremental garbage collection
        for j in range(len(first_past_key_values)):
            
            # kv buffers with the new shapes (w/ fill batch size)
            padded_past_keys = first_past_key_values[j][0].new_zeros(padded_past_shape)
            padded_past_values = first_past_key_values[j][1].new_zeros(padded_past_shape)
            
            # fill the buffers with the kvs from each batch
            start_index = 0
            for batch in batches:
                end_index = start_index + len(batch)

                # extract past kvs, clearing references for gc
                past_keys, past_values = batch.past_key_values[j]
                batch.past_key_values[j][0] = None 
                batch.past_key_values[j][1] = None 

                # copy into the buffer, padding to the left
                past_seq_len = batch.max_input_length - 1
                assert past_seq_len == past_keys.shape[2] and past_seq_len == past_values.shape[2]
                
                padded_past_keys[
                    start_index:end_index, :, -past_seq_len:, :
                ] = past_keys
                
                padded_past_values[
                    start_index:end_index,:, -past_seq_len:, :
                ] = past_values
                
                # free gpu memory
                del past_keys
                del past_values

                start_index = end_index

            past_key_values.append([padded_past_keys, padded_past_values])

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            lora_ids=lora_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            stopping_criterias=stopping_criterias,
            input_lengths=input_lengths,
            max_input_length=max_input_length,
            padding_right_offset=padding_right_offset
        )

class BLoraCausalLM:
    def __init__(
        self,
        base_model_id: str,
        lora_ids: List[str],
        has_position_ids: bool = True,
        dtype: torch.dtype = torch.float16,
        blinear_type: Optional[str] = None,
    ):
        self.active_batch_id = None
        self.blinear_type = blinear_type
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto", 
            torch_dtype=dtype
        )

        self.model, self.lora_map = load_loras(self.model, lora_ids, blinear_type=blinear_type)
        self.has_position_ids = has_position_ids

        # self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_id)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None:
            assert self.tokenizer.eos_token_id is not None
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def unset_batch_lora_ids(self):
        self.active_batch_id = None
        
    def set_batch_lora_ids(
        self,
        batch_id: int,
        batch_lora_ids: List[str],
        is_prefill: bool = False
    ):  
        # if prefill --> then the lora_ids need to be reset
        #       note: if prefill, leave active_batch as none
        #             prefills are followed by concatenate, which will invalidate
        
        if (is_prefill or 
            self.active_batch_id is None or 
            self.active_batch_id != batch_id
        ):
            # set active_batch_id so we do not need to call this again
            self.active_batch_id = batch_id if not is_prefill else None

            # TODO (@rsnm2): figure out if this take a long time and make it faster
            inp_loras = [self.lora_map[lora_id] for lora_id in batch_lora_ids]
            for _, module in self.model.named_modules():
                if self.blinear_type == "bmm":
                    if hasattr(module, "set_batch_lora_ids"):
                        module.set_batch_lora_ids(inp_loras)
                else:
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

    def generate_token(
        self,
        batch: BLoraCausalLMBatch,
    ) -> (List[Generation], Optional[BLoraCausalLMBatch]):
        
        # set loras to match the current batch
        self.set_batch_lora_ids(
            batch_id=batch.batch_id, 
            batch_lora_ids=batch.lora_ids, 
            is_prefill=batch.past_key_values is None
        )

        # run forward pass, filtering the attn mask
        offset = -batch.padding_right_offset if batch.padding_right_offset > 0 else batch.attention_mask.shape[1]
        logits, past = self.forward(
            batch.input_ids,
            batch.attention_mask[:,:offset],
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
            batch.stopping_criterias,
            logits,
        )
        for i, (
            request, 
            input_length,
            stopping_criteria,
            logits,
        ) in enumerate(iterator):
            
            # (a) sample
            next_token_id = self.greedy(logits)
            new_input_length = input_length + 1

            # (b) check stopping
            stop, finish_reason = stopping_criteria(token_id=next_token_id)
            if not stop:
                all_stopped = False
            else:
                
                self.unset_batch_lora_ids()
  
            # c) make generation
            generations.append(Generation(
                request_id=request.id,
                token_id=next_token_id.item(),
                stopped=stop,
                finish_reason=finish_reason
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