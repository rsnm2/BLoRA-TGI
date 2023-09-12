import fastapi, uvicorn
from sse_starlette.sse import EventSourceResponse
from contextlib import asynccontextmanager

from threading import Thread
from router import TextGenerationRouter, batching_task
from utils import GenerateRequestInputs, GenerateRequestOutputs, GenerateRequest

MESSAGE_STREAM_RETRY_TIMEOUT = 15000  # milisecond

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

@app.post("/generate_stream")
async def generate_stream_stream(request: fastapi.Request, inputs: GenerateRequestInputs):
    # convert input to generate request, submit to router
    generate_request = GenerateRequest.from_gr_inputs(inputs)
    artifacts["router"].submit_request(generate_request)

    async def token_generator():
        generation = generate_request.response_stream.get()    
        
        while not generation.stopped:
            if await request.is_disconnected():
                break

            yield artifacts["router"].service.model.tokenizer.decode(generation.token_id)
            generation = generate_request.response_stream.get()

    return EventSourceResponse(token_generator())

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5543,
        workers=1,      # limit to one process to avoid copying the model
    )