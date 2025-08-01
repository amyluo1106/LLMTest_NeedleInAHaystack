import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import List, Optional, Union, Dict, Any, AsyncGenerator

import asyncio
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder

# LMCache configuration - make sure these are set before any LMCache imports
BLEND_SPECIAL_STR = " # # "
os.environ.update({
    "LMCACHE_CHUNK_SIZE": "20736",
    "LMCACHE_ENABLE_BLENDING": "True",
    "LMCACHE_BLEND_SPECIAL_STR": BLEND_SPECIAL_STR,
    "LMCACHE_USE_LAYERWISE": "True",
    "LMCACHE_LOCAL_CPU": "True",
    "LMCACHE_MAX_LOCAL_CPU_SIZE": "40",  # GiB
    "LMCACHE_BLEND_RECOMPUTE_RATIO": "0.15",
    "LMCACHE_BLEND_MIN_TOKENS": "512"
})

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
LMCACHE_CONNECTOR = "LMCacheConnectorV1"

# Tokenizer cache
_tokenizer = None
_tok_lock = threading.Lock()

def get_tokenizer():
    global _tokenizer
    with _tok_lock:
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return _tokenizer

# OpenAI API compatible models
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 256
    stream: bool = False

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 256
    stream: bool = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{int(time.time())}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: ChatCompletionUsage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "organization-owner"
    
class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# Helper for the original /generate endpoint
class GenRequest(BaseModel):
    prompt: Union[List[int], str]
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 10
    req_str: str = "request"

class GenResponse(BaseModel):
    texts: List[str]
    generation_time: float
    req_str: str

def _build_llm():
    ktc = KVTransferConfig(kv_connector=LMCACHE_CONNECTOR, kv_role="kv_both")
    
    # Use a single definition of parameters
    return LLM(
        model=MODEL_NAME,
        kv_transfer_config=ktc,
        max_model_len=45000,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        tensor_parallel_size=4,
        max_num_batched_tokens=41000, # ??????? who added this
        enforce_eager=True,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Initializing LLM with LMCache ({LMCACHE_CONNECTOR})...")
    tokenizer = get_tokenizer()
    llm = _build_llm()
    app.state.tokenizer = tokenizer
    app.state.llm = llm
    print("LLM initialization complete.")
    yield
    print("Shutting down LMCache engine...")
    # graceful shutdown
    LMCacheEngineBuilder.destroy(ENGINE_NAME)

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to count tokens
def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))

# Helper to check if text contains blend separators
def contains_blend_separators(text: str) -> bool:
    return BLEND_SPECIAL_STR in text

# Original /generate endpoint - this supports the explicit token IDs input
# needed for optimal LMCache performance
@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest):
    if isinstance(req.prompt, str):
        prompt_ids = app.state.tokenizer.encode(req.prompt)
    else:
        prompt_ids = req.prompt

    sp = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )

    start = time.time()
    outputs = app.state.llm.generate(prompt_token_ids=prompt_ids, sampling_params=sp)
    texts = [o.outputs[0].text for o in outputs]
    elapsed = time.time() - start

    print("-" * 50)
    for t in texts:
        print(f"Generated text: {t!r}")
    print(f"Generation took {elapsed:.2f} s, {req.req_str} request done.")
    print("-" * 50, flush=True)

    return GenResponse(texts=texts, generation_time=elapsed, req_str=req.req_str)

# Format chat messages for Llama models
def format_chat_prompt(messages):
    formatted_parts = []
    
    for i, msg in enumerate(messages):
        if msg.role == "system":
            formatted_parts.append(f"<|system|>\n{msg.content}")
        elif msg.role == "user":
            formatted_parts.append(f"<|user|>\n{msg.content}")
        elif msg.role == "assistant":
            formatted_parts.append(f"<|assistant|>\n{msg.content}")
        else:
            # Handle any other roles as user messages
            formatted_parts.append(f"<|user|>\n{msg.content}")
    
    # Add the final assistant marker for generation
    if not (messages and messages[-1].role == "assistant"):
        formatted_parts.append("<|assistant|>")
    
    # Join all parts with newlines
    formatted_prompt = "\n".join(formatted_parts)
    
    return formatted_prompt

# OpenAI API compatible endpoints
@app.get("/v1/models")
async def list_models():
    """Get available models"""
    models = [ModelInfo(id=MODEL_NAME)]
    return ModelList(data=models)

# @app.post("/v1/chat/completions")
# async def create_chat_completion(request: ChatCompletionRequest):
#     """OpenAI-compatible chat completions endpoint"""
#     print(f"Chat completion request received for model: {request.model}")
    
#     try:
#         # Format the chat prompt for our model
#         prompt_text = format_chat_prompt(request.messages)
        
#         # print("TESTING)")
#         # print(prompt_text)

#         # Special handling for user content that might contain LMCache blend separators
#         user_messages = [m for m in request.messages if m.role == "user"]
#         has_blend_separators = any(contains_blend_separators(m.content) for m in user_messages)
        
#         # Count tokens for usage info
#         prompt_tokens = count_tokens(prompt_text, app.state.tokenizer)
        
#         # Convert text to token IDs - this is crucial for LMCache to work correctly
#         prompt_token_ids = app.state.tokenizer.encode(prompt_text)
        
#         # Set up sampling parameters
#         sp = SamplingParams(
#             temperature=request.temperature,
#             top_p=request.top_p,
#             max_tokens=request.max_tokens,
#         )
        
#         # Handle streaming response
#         if request.stream:
#             async def stream_response():
#                 id_str = f"chatcmpl-{uuid.uuid4().hex}"
#                 start_time = time.time()
                
#                 # Start the stream
#                 yield f"data: {{\n"
#                 yield f'"id": "{id_str}",\n'
#                 yield f'"object": "chat.completion.chunk",\n'
#                 yield f'"created": {int(start_time)},\n'
#                 yield f'"model": "{request.model}",\n'
#                 yield '"choices": [{\n'
#                 yield '"index": 0,\n'
#                 yield '"delta": {"role": "assistant"},\n'
#                 yield '"finish_reason": null\n'
#                 yield "}]\n"
#                 yield "}\n\n"
                
#                 # Generate the completion
#                 outputs = app.state.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sp)
#                 generated_text = outputs[0].outputs[0].text
                
#                 # Stream the response token by token (simplified)
#                 for i, char in enumerate(generated_text):
#                     yield f"data: {{\n"
#                     yield f'"id": "{id_str}",\n'
#                     yield f'"object": "chat.completion.chunk",\n'
#                     yield f'"created": {int(start_time)},\n'
#                     yield f'"model": "{request.model}",\n'
#                     yield '"choices": [{\n'
#                     yield '"index": 0,\n'
#                     yield f'"delta": {{"content": "{char}"}},\n'
#                     yield '"finish_reason": null\n'
#                     yield "}]\n"
#                     yield "}\n\n"
                    
#                     # Small delay to simulate token-by-token streaming
#                     await asyncio.sleep(0.01)
                
#                 # Send the final chunk
#                 yield f"data: {{\n"
#                 yield f'"id": "{id_str}",\n'
#                 yield f'"object": "chat.completion.chunk",\n'
#                 yield f'"created": {int(start_time)},\n'
#                 yield f'"model": "{request.model}",\n'
#                 yield '"choices": [{\n'
#                 yield '"index": 0,\n'
#                 yield '"delta": {},\n'
#                 yield '"finish_reason": "stop"\n'
#                 yield "}]\n"
#                 yield "}\n\n"
                
#                 # End the stream
#                 yield "data: [DONE]\n\n"
            
#             return StreamingResponse(stream_response(), media_type="text/event-stream")
        
#         else:
#             # Generate completion
#             start_time = time.time()
            
#             outputs = app.state.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sp)
#             generated_text = outputs[0].outputs[0].text
#             print(f"Generated chat response: {generated_text}")
            
#             # Count completion tokens
#             completion_tokens = count_tokens(generated_text, app.state.tokenizer)
            
#             # Create response
#             choice = ChatCompletionChoice(
#                 index=0,
#                 message=ChatMessage(role="assistant", content=generated_text),
#                 finish_reason="stop"
#             )
            
#             usage = ChatCompletionUsage(
#                 prompt_tokens=prompt_tokens,
#                 completion_tokens=completion_tokens,
#                 total_tokens=prompt_tokens + completion_tokens
#             )
            
#             return ChatCompletionResponse(
#                 model=request.model,
#                 choices=[choice],
#                 usage=usage
#             )
#     except Exception as e:
#         print(f"Error generating completion: {str(e)}")
#         import traceback
#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")

def format_llama3_chat_prompt(messages):
    """Format chat messages for Llama 3.1 model"""
    formatted_parts = []
    
    for msg in messages:
        if msg.role == "system":
            formatted_parts.append(f"<|start_header_id|>system<|end_header_id|>\n{msg.content}")
        elif msg.role == "user":
            formatted_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{msg.content}")
        elif msg.role == "assistant":
            formatted_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{msg.content}")
    
    # Add the assistant marker for generation
    if not (messages and messages[-1].role == "assistant"):
        formatted_parts.append("<|start_header_id|>assistant<|end_header_id|>")
    
    # Join with newlines
    return "\n".join(formatted_parts)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint using the same logic as /generate"""
    print(f"Chat completion request received for model: {request.model}")
    
    try:
        # Format the chat messages into Llama 3.1 format
        formatted_prompt = format_llama3_chat_prompt(request.messages)
        
        # Convert to token IDs - exactly like in generate
        prompt_token_ids = app.state.tokenizer.encode(formatted_prompt)
        
        # Set up sampling parameters - exactly like in generate
        sp = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=2*request.max_tokens,
        )
        
        # Generate completion - exactly like in generate
        start = time.time()
        outputs = app.state.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sp)
        texts = [o.outputs[0].text for o in outputs]
        elapsed = time.time() - start
        
        # Debug output - exactly like in generate
        print("-" * 50)
        for t in texts:
            print(f"Generated text: {t!r}")
        print(f"Generation took {elapsed:.2f} s")
        print("-" * 50, flush=True)
        
        # Get the generated text
        generated_text = texts[0] if texts else ""
        
        # Count tokens for usage info
        prompt_tokens = len(prompt_token_ids)
        completion_tokens = count_tokens(generated_text, app.state.tokenizer)
        
        # Create the OpenAI-compatible response object
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=generated_text),
            finish_reason="stop"
        )
        
        usage = ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        
        return ChatCompletionResponse(
            model=request.model,
            choices=[choice],
            usage=usage
        )
        
    except Exception as e:
        print(f"Error generating completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")


# @app.post("/v1/chat/completions")
# async def create_chat_completion(request: ChatCompletionRequest):
#     """OpenAI-compatible chat completions endpoint for Llama 3.1"""
#     print(f"Chat completion request received for model: {request.model}")
    
#     try:
#         # Special detection for needle-in-haystack format
#         is_retrieval_task = False
        
#         # Check if this matches the expected format for needle-in-haystack
#         if (len(request.messages) >= 3 and 
#             request.messages[0].role == "system" and
#             all(msg.role == "user" for msg in request.messages[1:3])):
            
#             is_retrieval_task = True
#             print("Detected needle-in-haystack retrieval task")
        
#         # Format the prompt using Llama 3.1 specific formatter
#         prompt_text = format_chat_prompt(request.messages)
        
#         # Log the formatted prompt for debugging
#         print(f"Formatted Llama 3.1 prompt:\n{prompt_text[:200]}...")
        
#         # Count tokens for usage info
#         prompt_tokens = count_tokens(prompt_text, app.state.tokenizer)
        
#         # Convert text to token IDs
#         prompt_token_ids = app.state.tokenizer.encode(prompt_text)
        
#         # Set up sampling parameters
#         sp = SamplingParams(
#             temperature=request.temperature,
#             top_p=request.top_p,
#             max_tokens=request.max_tokens,
#             stop=["<|reserved_special_token", "<|im_end|>", "<|endoftext|>", "</s>"]
#         )
        
#         # Generate completion
#         start_time = time.time()
#         outputs = app.state.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sp)
#         generated_text = outputs[0].outputs[0].text
#         generation_time = time.time() - start_time
        
#         # Log response info
#         if is_retrieval_task:
#             print(f"Retrieval task completed in {generation_time:.2f}s. Response: {generated_text[:100]}...")
#         else:
#             print(f"Generated chat response in {generation_time:.2f}s: {generated_text[:100]}...")
        
#         # Count completion tokens
#         completion_tokens = count_tokens(generated_text, app.state.tokenizer)
        
#         # Create response
#         choice = ChatCompletionChoice(
#             index=0,
#             message=ChatMessage(role="assistant", content=generated_text),
#             finish_reason="stop"
#         )
        
#         usage = ChatCompletionUsage(
#             prompt_tokens=prompt_tokens,
#             completion_tokens=completion_tokens,
#             total_tokens=prompt_tokens + completion_tokens
#         )
        
#         return ChatCompletionResponse(
#             model=request.model,
#             choices=[choice],
#             usage=usage
#         )
#     except Exception as e:
#         print(f"Error generating completion: {str(e)}")
#         import traceback
#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")


# Add health endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Add ping endpoint for compatibility
@app.get("/ping")
@app.post("/ping")
async def ping():
    return {"status": "ok"}

# Add version endpoint
@app.get("/version")
async def version():
    return {"version": "1.0.0"}

# Add ChatSession API for compatibility with existing code
class ChatSession:
    def __init__(self, port: int = 8000, blend_special_str: str = BLEND_SPECIAL_STR):
        self.sep_ids = get_tokenizer().encode(blend_special_str)[1:]
        self.temperature = 0.0
        self.context_ids = []
        self.context = ""
        self.url = f"http://localhost:{port}/generate"

    def set_context(self, context_list):
        tok = get_tokenizer()
        if not context_list:
            self.context_ids = []
            return
        
        # build context string
        self.context = context_list[0]
        for chunk in context_list[1:]:
            self.context += BLEND_SPECIAL_STR + chunk

        ids = tok.encode(context_list[0])
        for chunk in context_list[1:]:
            ids += self.sep_ids + tok.encode(chunk)[1:]
        self.context_ids = ids

    def get_context(self):
        return self.context

    def chat(self, question: str):
        tok = get_tokenizer()
        prompt_ids = (
            self.context_ids +
            self.sep_ids +
            tok.encode(question)[1:]
        )

        payload = {
            "prompt": prompt_ids,
            "temperature": self.temperature,
            "top_p": 0.95,
            "max_tokens": 512,
            "req_str": "chat",
        }

        t0 = time.time()
        r = requests.post(self.url, json=payload, timeout=120)
        t1 = time.time()

        r.raise_for_status()
        data = r.json()

        gen_time = data.get("generation_time", 0.0)
        total_time = t1 - t0
        overhead = total_time - gen_time
        text = data.get("texts", [""])[0]

        print("-" * 60)
        print(f"Request 'chat'")
        print(f"Generated text          : {text!r}")
        print(f"Server generation_time  : {gen_time:.2f} s")
        print(f"Client round-trip time  : {total_time:.2f} s")
        print(f"⇢ Network / overhead    : {overhead:.2f} s")
        print("-" * 60, flush=True)

        yield text
        yield f"\n\n(TTFT: {gen_time:.2f} s)"

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("lmcache_openai_server:app", host="0.0.0.0", port=8000, log_level="info")






# import os
# BLEND_SPECIAL_STR = " # # "
# os.environ.update(
#     {
#         "LMCACHE_CHUNK_SIZE": "256",
#         "LMCACHE_ENABLE_BLENDING": "True",
#         "LMCACHE_BLEND_SPECIAL_STR": BLEND_SPECIAL_STR,
#         "LMCACHE_USE_LAYERWISE": "True",
#         "LMCACHE_LOCAL_CPU": "True",           
#         "LMCACHE_MAX_LOCAL_CPU_SIZE": "5",     # GiB
#         # "LMCACHE_LOCAL_DISK": "file://local_disk/",
#         # "LMCACHE_MAX_LOCAL_DISK_SIZE": "10",
#     }
# )

# from contextlib import asynccontextmanager
# from dataclasses import asdict
# import time

# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams
# from vllm.config import KVTransferConfig
# from vllm.engine.arg_utils import EngineArgs

# from lmcache.integration.vllm.utils import ENGINE_NAME
# from lmcache.v1.cache_engine import LMCacheEngineBuilder

# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# LMCACHE_CONNECTOR = "LMCacheConnectorV1"

# def _build_llm():
#     ktc = KVTransferConfig(kv_connector=LMCACHE_CONNECTOR, kv_role="kv_both")
#     engine_cfg = EngineArgs(
#         model=MODEL_NAME,
#         kv_transfer_config=ktc,
#         max_model_len=30000,
#         gpu_memory_utilization=0.9,
#         enable_prefix_caching=False,
#         max_num_batched_tokens=20480,
#         enforce_eager=True,
#     )
#     return LLM(**asdict(engine_cfg))

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     llm = _build_llm()
#     app.state.tokenizer = tokenizer
#     app.state.llm = llm
#     yield
#     # graceful shutdown
#     LMCacheEngineBuilder.destroy(ENGINE_NAME)

# app = FastAPI(lifespan=lifespan)

# class GenRequest(BaseModel):
#     prompt: list[int] | str          # allow raw text OR token-ids
#     temperature: float = 0.0
#     top_p: float = 0.95
#     max_tokens: int = 10
#     req_str: str = "request"

# class GenResponse(BaseModel):
#     texts: list[str]
#     generation_time: float
#     req_str: str

# @app.post("/generate", response_model=GenResponse)
# def generate(req: GenRequest):
#     if isinstance(req.prompt, str):
#         prompt_ids = app.state.tokenizer.encode(req.prompt)
#     else:
#         prompt_ids = req.prompt

#     sp = SamplingParams(
#         temperature=req.temperature,
#         top_p=req.top_p,
#         max_tokens=req.max_tokens,
#     )

#     start = time.time()
#     outputs = app.state.llm.generate(prompt_token_ids=prompt_ids, sampling_params=sp)
#     texts = [o.outputs[0].text for o in outputs]
#     elapsed = time.time() - start

#     print("-" * 50)
#     for t in texts:
#         print(f"Generated text: {t!r}")
#     print(f"Generation took {elapsed:.2f} s, {req.req_str} request done.")
#     print("-" * 50, flush=True)

#     return GenResponse(texts=texts, generation_time=elapsed, req_str=req.req_str)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("lmcache_openai_server:app", host="0.0.0.0", port=8000, log_level="info")




# CUDA_VISIBLE_DEVICES=0 python lmcache_openai_server.py \
#     --model "meta-llama/Llama-3.1-8B-Instruct" \
#     --lmcache-connector "LMCacheConnectorV1" \
#     --tensor-parallel-size 1 \
#     --port 8000



# export EVALUATOR_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# export OPENAI_API_BASE="http://localhost:8000/v1"
# export NIAH_MODEL_API_KEY="dummy-key"
# export NIAH_EVALUATOR_API_KEY="dummy-key"

# needlehaystack.run_test \
#     --provider openai \
#     --model_name "meta-llama/Llama-3.1-8B-Instruct" \
#     --document_depth_percents "[25]" \
#     --context_lengths "[2000]" \
#     --save_contexts "false"