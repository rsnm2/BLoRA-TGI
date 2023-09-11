import fastapi, uvicorn
from contextlib import asynccontextmanager

from threading import Thread
from router import TextGenerationRouter, batching_task
from utils import GenerateRequestInputs, GenerateRequestOutputs, GenerateRequest

base_model_id = "decapoda-research/llama-7b-hf"
lora_ids = ["jondurbin/airoboros-7b-gpt4-1.2-peft", "trl-lib/llama-7b-se-rl-peft", 'winddude/wizardLM-LlaMA-LoRA-7B']

artifacts = {}

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("\n--------------------       Building Router               --------------------\n")
    artifacts["router"] = TextGenerationRouter(
        base_model_id=base_model_id, 
        lora_ids=lora_ids
    )

    print("\n--------------------       Starting Batching Task        --------------------\n")
    batching_thread = Thread(
        target=batching_task,
        args=[artifacts["router"]]
    )
    batching_thread.start()

    print("\n--------------------       Launching App                 --------------------\n")
    yield
    
    print("\n--------------------       Shutting Down Batching Task   --------------------\n")
    artifacts["router"].stop_batching_task()
    batching_thread.join()

app = fastapi.FastAPI(lifespan=lifespan)

@app.post("/generate")
def generate(inputs: GenerateRequestInputs) -> GenerateRequestOutputs:

    # convert input to generate request
    generate_request = GenerateRequest.from_gr_inputs(inputs)

    # submit request to the router
    artifacts["router"].submit_request(generate_request)

    # build response
    tokens = []
    generation = generate_request.response_stream.get()
    while not generation.stopped:
        tokens.append(generation.token_id)
        generation = generate_request.response_stream.get()

    return GenerateRequestOutputs(
        response_text=artifacts["router"].service.model.tokenizer.decode(tokens),
        finish_reason=generation.finish_reason
    )

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5543,
        workers=1,      # limit to one process to avoid copying the model
        # reload=True
    )
