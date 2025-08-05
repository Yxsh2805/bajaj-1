import os
import time
import logging
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document as DocxDocument
import email
import io
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# RAG imports
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

class ReliabilityOptimizedVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_reliable(self, documents: List[Document]):
        """Optimized for reliability with Together.ai API"""
        logger.info(f"RELIABILITY: Processing {len(documents)} chunks")
        
        start_time = time.time()
        texts = [doc.page_content for doc in documents]
        
        # Filter out empty texts to prevent 400 errors
        valid_texts = []
        valid_docs = []
        for i, text in enumerate(texts):
            if text.strip() and len(text) > 10:  # Minimum length check
                valid_texts.append(text)
                valid_docs.append(documents[i])
            else:
                logger.warning(f"Skipping empty/short chunk: {text[:50]}...")
        
        if not valid_texts:
            logger.error("No valid texts to embed")
            return
        
        try:
            # Batch embedding for efficiency (Together.ai supports batch)
            vectors = self.embeddings.embed_documents(valid_texts)
            
            # Store results
            for doc, vector in zip(valid_docs, vectors):
                self.documents.append(doc)
                self.vectors.append(vector)
            
            success_rate = (len(vectors) / len(documents)) * 100
            logger.info(f"RELIABILITY: Embedded {len(vectors)}/{len(documents)} chunks in {time.time()-start_time:.1f}s ({success_rate:.1f}% success)")
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Fallback: Individual embedding with retries
            self._fallback_embedding(valid_docs)

    def _fallback_embedding(self, documents: List[Document]):
        """Fallback for when batch embedding fails"""
        logger.warning("Using fallback embedding method")
        for i, doc in enumerate(documents):
            try:
                vector = self.embeddings.embed_query(doc.page_content)
                self.documents.append(doc)
                self.vectors.append(vector)
                # Add delay to avoid rate limiting
                if i % 5 == 0:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Failed to embed chunk {i}: {str(e)[:60]}")

    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        """Efficient similarity search with error handling"""
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
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self.documents[:k] if self.documents else []

def reliable_document_loader(url: str) -> List[Document]:
    """Optimized loader for ≤25 page documents"""
    try:
        response = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'pdf' in content_type or url.lower().endswith('.pdf'):
            return _load_pdf(response.content, url)
        elif 'wordprocessingml' in content_type or url.lower().endswith('.docx'):
            return _load_docx(response.content, url)
        else:
            return _load_html(response.content, url)
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise HTTPException(status_code=400, detail=f"Document loading failed: {str(e)}")

def _load_pdf(content: bytes, url: str) -> List[Document]:
    """Load all pages for PDFs ≤25 pages"""
    pdf_file = io.BytesIO(content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    total_pages = len(pdf_reader.pages)
    
    # Process all pages (≤25 pages)
    text = ""
    for i in range(total_pages):
        page_text = pdf_reader.pages[i].extract_text()
        if page_text.strip():
            text += f"Page {i+1}: {page_text}\n\n"
    
    logger.info(f"Loaded PDF with {total_pages} pages")
    return [Document(page_content=text.strip(), metadata={"source": url, "type": "pdf"})]

def _load_docx(content: bytes, url: str) -> List[Document]:
    """Load DOCX documents"""
    docx_file = io.BytesIO(content)
    doc = DocxDocument(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return [Document(page_content=text.strip(), metadata={"source": url, "type": "docx"})]

def _load_html(content: bytes, url: str) -> List[Document]:
    """Load HTML content"""
    soup = BeautifulSoup(content, 'html.parser')
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    text = '\n'.join(line for line in lines if line)
    return [Document(page_content=text[:80000], metadata={"source": url, "type": "html"})]

class ReliabilityOptimizedRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing RELIABILITY OPTIMIZED RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            # More reliable embedding model
            self.embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
            
            # More reliable chat model with faster response
            self.chat_model = ChatTogether(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=0.2,
                max_tokens=2000
            )

            # Optimized chunking for reliability
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=120,
                separators=["\n\n", "\n", ". ", "! ", "? "]
            )

            self.initialized = True
            logger.info("RELIABILITY OPTIMIZED RAG engine ready!")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _reliable_query(self, vectorstore: ReliabilityOptimizedVectorStore, questions: List[str]) -> List[str]:
        """More reliable query handling with fallback"""
        try:
            # Try batch query first
            return self._batch_query(vectorstore, questions)
        except Exception as e:
            logger.error(f"Batch query failed: {e}, using fallback")
            return self._fallback_query(vectorstore, questions)

    def _batch_query(self, vectorstore: ReliabilityOptimizedVectorStore, questions: List[str]) -> List[str]:
        """Batch processing with better error handling"""
        context = ""
        for i, q in enumerate(questions[:3]):  # Get context from first 3 questions
            docs = vectorstore.similarity_search(q, k=2)
            context += f"Question {i+1}: {q}\nContext: {' '.join(d.page_content for d in docs)[:1500]}\n\n"
        
        system_content = """You are an insurance policy expert. Answer based ONLY on the context.
IMPORTANT:
- Answer in the same order as questions
- Separate answers with " | "
- Be concise (1-2 sentences)
- If unsure: "Info not found"
- Include key numbers and terms when available"""

        question_str = " | ".join(questions)
        prompt = f"Context:\n{context}\n\nQuestions:\n{question_str}\n\nAnswers:"

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=prompt)
        ]
        
        response = self.chat_model.invoke(messages)
        return self._parse_answers(response.content, len(questions))

    def _fallback_query(self, vectorstore: ReliabilityOptimizedVectorStore, questions: List[str]) -> List[str]:
        """Fallback when batch query fails"""
        answers = []
        for q in questions:
            try:
                docs = vectorstore.similarity_search(q, k=4)
                context = ' '.join(d.page_content for d in docs)[:2000]
                
                prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer concisely:"
                response = self.chat_model.invoke(prompt)
                answers.append(response.content.strip())
            except Exception:
                answers.append("Info not found")
        return answers

    def _parse_answers(self, response: str, expected_count: int) -> List[str]:
        """Robust answer parsing with fallback"""
        # First try standard separator
        if " | " in response:
            parts = response.split(" | ")
            if len(parts) == expected_count:
                return [p.strip() for p in parts]
        
        # Fallback separators
        for sep in ["\n", ";", "||", "//"]:
            if sep in response:
                parts = response.split(sep)
                if len(parts) == expected_count:
                    return [p.strip() for p in parts]
        
        # Final fallback - return as single answer array
        return [response.strip()] * expected_count

    async def process_reliable(self, url: str, questions: List[str]) -> List[str]:
        """Reliability-optimized processing with strict timeout"""
        if not self.initialized:
            self.initialize()
            
        try:
            # Enforce strict 25s timeout (5s buffer)
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout exceeded, returning fallback answers")
            return ["Info not found - timeout"] * len(questions)

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("Using cached vectorstore")
        else:
            try:
                docs = reliable_document_loader(url)
                chunks = self.text_splitter.split_documents(docs)
                
                # Enforce chunk limit (45 chunks max)
                if len(chunks) > 45:
                    chunks = chunks[:45]
                    logger.info(f"Truncated to 45 chunks (from {len(chunks)})")
                
                vectorstore = ReliabilityOptimizedVectorStore(self.embeddings)
                vectorstore.add_documents_reliable(chunks)
                self.vectorstore_cache[url_hash] = vectorstore
                logger.info(f"Cached vectorstore for {url_hash}")
            except Exception as e:
                logger.error(f"Vectorstore creation failed: {e}")
                # Return fallback answers if we can't process document
                return ["Info not found - processing error"] * len(questions)
        
        try:
            start_time = time.time()
            answers = self._reliable_query(vectorstore, questions)
            logger.info(f"LLM processed in {time.time()-start_time:.1f}s")
            return answers
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return ["Info not found - query error"] * len(questions)

# Global engine
rag_engine = ReliabilityOptimizedRAGEngine()

def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_engine.initialize()
    yield

app = FastAPI(title="RELIABILITY OPTIMIZED RAG API", version="4.0", lifespan=lifespan)

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
        
        answers = await rag_engine.process_reliable(request.documents, request.questions)
        return {"answers": answers}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error")
        return {"answers": ["Processing error"] * len(request.questions)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cache_entries": len(rag_engine.vectorstore_cache),
        "mode": "reliability_optimized",
        "timeout": "25s",
        "max_chunks": 45
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=30)
