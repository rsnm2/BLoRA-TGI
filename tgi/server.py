import fastapi, uvicorn
from contextlib import asynccontextmanager

from threading import Thread
from router import TextGenerationRouter, batching_task
from utils import GenerateRequestInputs, GenerateRequestOutputs, GenerateRequest

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base-model-id", type=str, required=True)
parser.add_argument("--lora-ids", nargs='+', type=str, required=True)

args = parser.parse_args()
base_model_id = args.base_model_id
lora_ids = args.lora_ids

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
    
    gr_outputs = GenerateRequestOutputs()

    # build response
    generation = generate_request.response_stream.get()
    while not generation.stopped:
        gr_outputs.response_text += generation.token
        generation = generate_request.response_stream.get()

    gr_outputs.finish_reason = generation.finish_reason
    return gr_outputs

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5543,
        workers=1,      # limit to one process to avoid copying the model
        # reload=True
    )
