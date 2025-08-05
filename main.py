import os
import time
import logging
import hashlib
import numpy as np
import asyncio
import requests
import io
import PyPDF2
from typing import List, Optional, Dict, Any  # <-- Added all necessary typing imports
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage  # <-- Added import for HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

# --- Constants ---
EMBEDDING_MODELS = [
    "BAAI/bge-base-en-v1.5",  # Primary reliable model
    "togethercomputer/m2-bert-80M-8k-retrieval",  # Fallback 1
    "togethercomputer/m2-bert-80M-32k-retrieval"  # Fallback 2
]
CHAT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

class RobustVectorStore:
    def __init__(self):
        self.embeddings = None
        self.documents = []
        self.vectors = []
        self.initialize_embeddings()
    
    def initialize_embeddings(self):
        """Initialize embeddings with fallback models"""
        for model in EMBEDDING_MODELS:
            try:
                self.embeddings = TogetherEmbeddings(model=model)
                logger.info(f"Using embedding model: {model}")
                return
            except Exception as e:
                logger.warning(f"Model {model} failed: {str(e)[:60]}")
        
        raise RuntimeError("All embedding models failed to initialize")
    
    def add_documents(self, documents: List[Document]):
        logger.info(f"Processing {len(documents)} chunks")
        start_time = time.time()
        
        # Filter empty documents
        valid_docs = [doc for doc in documents if doc.page_content.strip()]
        
        # Batch processing with retries
        for attempt in range(3):
            try:
                texts = [doc.page_content for doc in valid_docs]
                vectors = self.embeddings.embed_documents(texts)
                
                # Store results
                for doc, vector in zip(valid_docs, vectors):
                    self.documents.append(doc)
                    self.vectors.append(vector)
                
                logger.info(f"Embedded {len(vectors)} chunks in {time.time()-start_time:.1f}s")
                return
            except Exception as e:
                logger.error(f"Batch embedding attempt {attempt+1} failed: {str(e)[:60]}")
                time.sleep(0.5 * (attempt + 1))
        
        # Fallback to individual embedding
        logger.warning("Using fallback individual embedding")
        for i, doc in enumerate(valid_docs):
            try:
                vector = self.embeddings.embed_query(doc.page_content)
                self.documents.append(doc)
                self.vectors.append(vector)
                if i % 5 == 0:
                    time.sleep(0.1)
            except Exception:
                logger.error(f"Failed to embed chunk {i}")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            query_norm = np.linalg.norm(query_vector)
            
            # Efficient numpy-based cosine similarity
            vectors_array = np.array(self.vectors)
            norms = np.linalg.norm(vectors_array, axis=1)
            valid_mask = (query_norm > 0) & (norms > 0)
            
            if np.any(valid_mask):
                similarities = np.zeros(len(self.vectors))
                valid_indices = np.where(valid_mask)[0]
                similarities[valid_indices] = np.dot(vectors_array[valid_indices], query_vector) / (norms[valid_indices] * query_norm)
                top_indices = np.argsort(similarities)[-k:][::-1]
                return [self.documents[i] for i in top_indices]
            return []
        except Exception:
            return self.documents[:k]

def load_document(url: str) -> str:
    """Simplified document loader focusing on text extraction"""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        if 'pdf' in response.headers.get('content-type', '').lower():
            return extract_pdf_text(response.content)
        return response.text[:100000]  # Limit HTML/text
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise HTTPException(status_code=400, detail=f"Document loading failed")

def extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF with PyPDF2"""
    try:
        pdf_file = io.BytesIO(content)
        reader = PyPDF2.PdfReader(pdf_file)
        return "\n\n".join(page.extract_text() for page in reader.pages)
    except Exception:
        return "PDF content could not be extracted"

class RobustRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.text_splitter = None
        self.vectorstore_cache = {}
        self.initialized = False
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing ROBUST RAG engine...")
        os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
        
        # Initialize chat model
        self.chat_model = ChatTogether(
            model=CHAT_MODEL,
            temperature=0.2,
            max_tokens=1500
        )

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". "]
        )
        
        self.initialized = True
        logger.info("RAG engine ready!")

    async def process(self, url: str, questions: List[str]) -> List[str]:
        if not self.initialized:
            self.initialize()
        
        try:
            # Strict timeout with buffer
            return await asyncio.wait_for(
                self._process(url, questions),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            return ["Info not found - timeout"] * len(questions)

    async def _process(self, url: str, questions: List[str]) -> List[str]:
        # Get or create vectorstore
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        vectorstore = self.vectorstore_cache.get(url_hash)
        
        if not vectorstore:
            text = load_document(url)
            docs = [Document(page_content=text, metadata={"source": url})]
            chunks = self.text_splitter.split_documents(docs)
            
            # Strict chunk limit
            if len(chunks) > 35:
                chunks = chunks[:35]
            
            vectorstore = RobustVectorStore()
            vectorstore.add_documents(chunks)
            self.vectorstore_cache[url_hash] = vectorstore
        
        # Generate answers with fallback
        try:
            return self._generate_answers(vectorstore, questions)
        except Exception:
            return ["Info not found - processing error"] * len(questions)

    def _generate_answers(self, vectorstore: RobustVectorStore, questions: List[str]) -> List[str]:
        """Generate answers with multiple fallback strategies"""
        # Try batch processing first
        try:
            return self._batch_query(vectorstore, questions)
        except Exception:
            return self._individual_query(vectorstore, questions)

    def _batch_query(self, vectorstore: RobustVectorStore, questions: List[str]) -> List[str]:
        context = ""
        for q in questions[:3]:  # Get context for first 3 questions
            docs = vectorstore.similarity_search(q, k=2)
            context += " ".join(d.page_content for d in docs)[:2000] + "\n\n"
        
        prompt = f"""Answer these questions based solely on the context:
Context: {context}

Questions: {" | ".join(questions)}

Instructions:
- Answer in the same order
- Separate answers with " | "
- Be concise (1-2 sentences)
- If answer unknown: "Info not found"
- Include key numbers when available"""

        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        return self._parse_answers(response.content, len(questions))

    def _individual_query(self, vectorstore: RobustVectorStore, questions: List[str]) -> List[str]:
        """Fallback query method"""
        answers = []
        for q in questions:
            try:
                docs = vectorstore.similarity_search(q, k=3)
                context = " ".join(d.page_content for d in docs)[:1500]
                prompt = f"Context: {context}\n\nQuestion: {q}\nAnswer concisely:"
                response = self.chat_model.invoke([HumanMessage(content=prompt)])
                answers.append(response.content.strip())
            except Exception:
                answers.append("Info not found")
        return answers

    def _parse_answers(self, response: str, expected_count: int) -> List[str]:
        # Multiple parsing strategies
        for sep in [" | ", "\n", ";", "||"]:
            if sep in response:
                parts = response.split(sep)
                if len(parts) == expected_count:
                    return [p.strip() for p in parts]
        
        # Fallback: return first N sentences
        sentences = [s.strip() for s in response.split(". ") if s.strip()]
        return sentences[:expected_count]

# Global engine
rag_engine = RobustRAGEngine()

def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

app = FastAPI(title="ROBUST RAG API", version="5.0")

@app.on_event("startup")
async def startup_event():
    rag_engine.initialize()

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    try:
        logger.info(f"Processing {len(request.questions)} questions")
        
        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        answers = await rag_engine.process(request.documents, request.questions)
        return {"answers": answers}
    
    except HTTPException:
        raise
    except Exception:
        return {"answers": ["Processing error"] * len(request.questions)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cache_entries": len(rag_engine.vectorstore_cache),
        "embedding_models": EMBEDDING_MODELS,
        "chat_model": CHAT_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=30)
