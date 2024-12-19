from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi import Query
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Any
from decouple import config as _config
import sys
import os
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from pathlib import Path
from typing import Annotated, List, Dict, Any
import asyncio
import nest_asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from lightrag.kg.oracle_impl import OracleDB

print(os.getcwd())
script_directory = Path(__file__).resolve().parent.parent
sys.path.append(os.path.abspath(script_directory))


# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

DEFAULT_RAG_DIR = "index_default"


# We use OpenAI compatible API to call LLM on Oracle Cloud
# More docs here https://github.com/jin38324/OCI_GenAI_access_gateway
BASE_URL = _config("OCI_LLM_API_URL")
APIKEY = _config("OCI_LLM_API_KEY")

# Configure working directory
WORKING_DIR = _config("WORKSPACE")
print(f"WORKING_DIR: {WORKING_DIR}")
LLM_MODEL = _config("OCI_LLM_MODEL")
print(f"LLM_MODEL: {LLM_MODEL}")
EMBEDDING_MODEL = _config("OCI_EMBEDDING_MODEL")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
EMBEDDING_MAX_TOKEN_SIZE = int(_config("EMBEDDING_MAX_TOKEN_SIZE"))
print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=APIKEY,
        base_url=BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model=EMBEDDING_MODEL,
        api_key=APIKEY,
        base_url=BASE_URL,
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


async def init():
    # Detect embedding dimension
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")
    # Create Oracle DB connection
    # The `config` parameter is the connection configuration of Oracle DB
    # More docs here https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html
    # We storage data in unified tables, so we need to set a `workspace` parameter to specify which docs we want to store and query
    # Below is an example of how to connect to Oracle Autonomous Database on Oracle Cloud

    oracle_db = OracleDB(
        config={
            "host": _config("HOST"),
            "port": int(_config("PORT")),
            "user": _config("PASSWORD"),
            "password": _config("PASSWORD"),
            "dsn": _config("DSN"),
            "workspace": _config("WORKSPACE"),
        }  # specify which docs you want to store and query
    )

    # Check if Oracle DB tables exist, if not, tables will be created
    await oracle_db.check_tables()
    # Initialize LightRAG
    # We use Oracle DB as the KV/vector/graph storage
    # You can add `addon_params={"example_number": 1, "language": "Simplfied Chinese"}` to control the prompt
    rag = LightRAG(
        enable_llm_cache=False,
        working_dir=WORKING_DIR,
        chunk_token_size=512,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=512,
            func=embedding_func,
        ),
        graph_storage="OracleGraphStorage",
        kv_storage="OracleKVStorage",
        vector_storage="OracleVectorDBStorage",
    )

    # Setthe KV/vector/graph storage's `db` property, so all operation will use same connection pool
    rag.graph_storage_cls.db = oracle_db
    rag.key_string_value_json_storage_cls.db = oracle_db
    rag.vector_db_storage_cls.db = oracle_db

    #rag.graph_storage_cls.__dict__['db'] = oracle_db
    #rag.key_string_value_json_storage_cls.__dict__['db']  = oracle_db
    #rag.vector_db_storage_cls.__dict__['db'] = oracle_db

    print(rag.chunk_entity_relation_graph.__dict__)

    return rag


# Extract and Insert into LightRAG storage
# with open("./dickens/book.txt", "r", encoding="utf-8") as f:
#   await rag.ainsert(f.read())

# # Perform search in different modes
# modes = ["naive", "local", "global", "hybrid"]
# for mode in modes:
#     print("="*20, mode, "="*20)
#     print(await rag.aquery("这篇文档是关于什么内容的?", param=QueryParam(mode=mode)))
#     print("-"*100, "\n")

# Data models


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    only_need_context: bool = False
    only_need_prompt: bool = False


class DataRequest(BaseModel):
    limit: int = 100


class InsertRequest(BaseModel):
    text: str


class Response(BaseModel):
    status: str
    data: Optional[Any] = None
    message: Optional[str] = None

class ChatRequest(BaseModel):
    message: str

# API routes

rag = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = await init()
    print(rag.graph_storage_cls.db)
    print(type(rag.graph_storage_cls.db))
    print("done!")
    yield


app = FastAPI(
    title="LightRAG API", description="API for RAG operations", lifespan=lifespan
)
# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", summary="default")
async def default(request: Request):
    return templates.TemplateResponse(
        request=request, name="chat.html"
    )
@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    # try:
    # loop = asyncio.get_event_loop()
    if request.mode == "naive":
        top_k = 3
    else:
        top_k = 60
    result = await rag.aquery(
        request.query,
        param=QueryParam(
            mode=request.mode,
            only_need_context=request.only_need_context,
            only_need_prompt=request.only_need_prompt,
            top_k=top_k,
        ),
    )
    return Response(status="success", data=result)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))


@app.get("/data", response_model=Response)
async def query_all_nodes(type: str = Query("nodes"), limit: int = Query(100)):

    if type == "nodes":
        result = await rag.chunk_entity_relation_graph.get_all_nodes(limit=limit)
    elif type == "edges":
        result = await rag.chunk_entity_relation_graph.get_all_edges(limit=limit)
    elif type == "statistics":
        result = await rag.chunk_entity_relation_graph.get_statistics()
    return Response(status="success", data=result)


@app.post("/insert", response_model=Response)
async def insert_endpoint(request: InsertRequest):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(request.text))
        return Response(status="success", message="Text inserted successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    print(request.message)
    print("chat")


@app.post("/upload")
async def upload(files: Annotated[list[bytes], File()]):
    for file in files:
        try:
            content = file.decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try other encodings
            content = file.decode("gbk")
        # Insert file content
        loop = asyncio.get_event_loop()
        #await loop.run_in_executor(None, lambda: rag.insert(content))
        await rag.ainsert(content)
        return Response(
            status="success",
            message=f"File content inserted successfully",
        )
@app.post("/insert_file", response_model=Response)
async def insert_file(file: UploadFile = File(...)):
    print(1)
    try:
        file_content = await file.read()
        # Read file content
        try:
            content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try other encodings
            content = file_content.decode("gbk")
        # Insert file content
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(content))

        return Response(
            status="success",
            message=f"File content from {file.filename} inserted successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8020)

# Usage example
# To run the server, use the following command in your terminal:
# python lightrag_api_openai_compatible_demo.py

# Example requests:
# 1. Query:
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "your query here", "mode": "hybrid"}'

# 2. Insert text:
# curl -X POST "http://127.0.0.1:8020/insert" -H "Content-Type: application/json" -d '{"text": "your text here"}'

# 3. Insert file:
# curl -X POST "http://127.0.0.1:8020/insert_file" -H "Content-Type: application/json" -d '{"file_path": "path/to/your/file.txt"}'

# 4. Health check:
# curl -X GET "http://127.0.0.1:8020/health"
