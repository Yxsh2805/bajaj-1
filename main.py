import os
import time
import logging
import hashlib
from typing import List, Optional
from contextlib import asynccontextmanager
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document as DocxDocument
import io
import numpy as np
import asyncio

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# LOCAL EMBEDDINGS - No API needed!
from sentence_transformers import SentenceTransformer

# Only Together.AI for chat (1 API call only)
from langchain_together import ChatTogether
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

class LocalVectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.documents = []
        self.vectors = []
    
    def add_documents_local(self, documents: List[Document]):
        """LOCAL EMBEDDING - No API calls, super fast!"""
        logger.info(f"LOCAL: Processing {len(documents)} chunks locally")
        
        start_time = time.time()
        
        # Prepare texts
        texts = [doc.page_content[:1600] for doc in documents]
        
        # LOCAL BATCH EMBEDDING - This is the magic!
        vectors = self.embedding_model.encode(
            texts,
            batch_size=32,  # Process in batches for speed
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        
        # Store results
        self.documents = documents
        self.vectors = vectors.tolist() if hasattr(vectors, 'tolist') else vectors
        
        embedding_time = time.time() - start_time
        logger.info(f"LOCAL: {embedding_time:.1f}s for {len(documents)} chunks - BLAZING FAST!")
    
    def similarity_search(self, query: str, k: int = 7) -> List[Document]:
        if not self.vectors:
            return []
        
        # Local query embedding
        query_vector = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        
        # Fast cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            sim = np.dot(query_vector, vector)  # Already normalized
            similarities.append((sim, i))
        
        # Sort and return top k
        similarities.sort(reverse=True)
        return [self.documents[i] for _, i in similarities[:k]]

def fast_document_loader(url: str) -> List[Document]:
    try:
        response = requests.get(url, timeout=6, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = min(len(pdf_reader.pages), 25)
            
            # Smart page selection
            if total_pages <= 10:
                pages_to_process = list(range(total_pages))
            else:
                # Strategic sampling
                first_pages = list(range(5))
                middle_pages = list(range(total_pages//2-2, total_pages//2+3))
                last_pages = list(range(total_pages-5, total_pages))
                pages_to_process = sorted(set(first_pages + middle_pages + last_pages))
            
            text_parts = []
            for i in pages_to_process:
                if i < len(pdf_reader.pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
            
            text = "\n\n".join(text_parts)
            logger.info(f"PDF: {len(pages_to_process)}/{total_pages} pages processed")
            return [Document(page_content=text, metadata={"source": url, "type": "pdf"})]
        
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            return [Document(page_content=text[:80000], metadata={"source": url, "type": "html"})]
        
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise

class LocalRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embedding_model = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing LOCAL RAG engine...")
        
        try:
            # LOCAL EMBEDDING MODEL - No API needed!
            logger.info("Loading local embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & good quality
            logger.info("Local embedding model ready!")
            
            # Only Together.AI for chat
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.chat_model = ChatTogether(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=0,
                max_tokens=2000
            )

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1400,
                chunk_overlap=120,
                separators=["\n\n", "\n", ". ", " "]
            )

            self.initialized = True
            logger.info("LOCAL RAG engine ready - NO API CALLS FOR EMBEDDINGS!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _local_query(self, vectorstore: LocalVectorStore, query: str) -> str:
        docs = vectorstore.similarity_search(query, k=7)
        context = " ".join([doc.page_content for doc in docs])[:2400]
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        system_content = """You are an insurance policy expert. Answer questions accurately.

RULES:
- Questions separated by " | ", answers separated by " | "
- DO NOT repeat questions
- Extract specific facts, numbers, conditions from document
- If not found: "Information not available"

Provide ONLY answers separated by " | "."""

        human_content = f"""Document: {context}

Questions: {query}

Answers (separated by " | "):"""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_local(self, url: str, questions: List[str]) -> List[str]:
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=22.0  # Even more aggressive
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="22 second timeout exceeded")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        total_start = time.time()
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED!")
        else:
            # Fast document processing
            docs = fast_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            # Limit chunks for speed
            if len(chunks) > 30:
                # Strategic selection
                start_count = 12
                middle_count = 8
                end_count = 10
                
                middle_start = len(chunks) // 2 - 4
                chunks = (chunks[:start_count] + 
                         chunks[middle_start:middle_start + middle_count] + 
                         chunks[-end_count:])
                
                logger.info(f"Selected 30/{len(chunks)} chunks")
            
            # LOCAL EMBEDDING - Super fast!
            vectorstore = LocalVectorStore(self.embedding_model)
            vectorstore.add_documents_local(chunks)
            
            self.vectorstore_cache[url_hash] = vectorstore
        
        # Query processing
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._local_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start
        logger.info(f"LOCAL: Query={query_time:.1f}s, Total={total_time:.1f}s")
        
        # Parse answers
        answers = [a.strip() for a in response.split(" | ") if a.strip() and len(a.strip()) > 5]
        
        while len(answers) < len(questions):
            answers.append("Information not available.")
        
        return answers[:len(questions)]

# Global engine
rag_engine = LocalRAGEngine()

def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rag_engine.initialize()
        logger.info("LOCAL RAG ready - No API dependencies!")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="LOCAL RAG API", version="LOCAL", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """LOCAL processing - no embedding API calls, super fast"""
    try:
        logger.info(f"LOCAL: {len(request.questions)} questions - TARGET: <20s")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_local(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"LOCAL: SUCCESS in {total_time:.1f}s - NO API BOTTLENECKS!")
        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

@app.get("/health")
async def health_check():
    return {
        "status": "local_ready",
        "embedding_provider": "Local SentenceTransformers",
        "model": "all-MiniLM-L6-v2",
        "api_calls": "only_chat_model",
        "target_time": "<20_seconds"
    }

@app.get("/")
async def root():
    return {"message": "LOCAL RAG API - No Embedding API Calls!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
