import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Union

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

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
        max_model_len=41001,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        tensor_parallel_size=4,
        max_num_batched_tokens=20480
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