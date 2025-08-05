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

class BalancedVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_balanced(self, documents: List[Document]):
        """Balanced approach - good accuracy, reasonable speed"""
        logger.info(f"BALANCED: Processing {len(documents)} chunks")
        start_time = time.time()
        
        # Clean texts but preserve content quality
        valid_texts = []
        valid_docs = []
        
        for doc in documents:
            text = doc.page_content.strip()
            # Light cleaning only
            text = text.replace('\x00', '').replace('\ufffd', '')
            
            if len(text) > 100:  # Meaningful content threshold
                valid_texts.append(text[:2000])  # Reasonable limit
                valid_docs.append(Document(page_content=text[:2000], metadata=doc.metadata))
        
        logger.info(f"BALANCED: {len(valid_texts)} valid chunks")
        
        # Simple individual embedding with rate limiting
        successful = 0
        for i, doc in enumerate(valid_docs):
            if successful >= 35:  # Good coverage limit
                break
                
            try:
                # Small delay every 5 chunks to prevent rate limiting
                if i > 0 and i % 5 == 0:
                    time.sleep(0.2)
                    
                vector = self.embeddings.embed_query(doc.page_content)
                self.documents.append(doc)
                self.vectors.append(vector)
                successful += 1
                
            except Exception as e:
                logger.warning(f"Embed failed for chunk {i}: {str(e)[:40]}")
                continue
        
        embedding_time = time.time() - start_time
        logger.info(f"BALANCED: {embedding_time:.1f}s for {successful} chunks")
    
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Proper cosine similarity for accuracy"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Proper cosine similarity
            similarities = []
            query_norm = np.linalg.norm(query_vector)
            
            for i, vector in enumerate(self.vectors):
                vector_norm = np.linalg.norm(vector)
                if query_norm > 0 and vector_norm > 0:
                    cos_sim = np.dot(query_vector, vector) / (query_norm * vector_norm)
                    similarities.append((cos_sim, i))
                else:
                    similarities.append((0.0, i))
            
            similarities.sort(reverse=True)
            return [self.documents[i] for _, i in similarities[:k]]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self.documents[:k] if len(self.documents) >= k else self.documents

def balanced_document_loader(url: str) -> List[Document]:
    """Balanced document loading"""
    try:
        logger.info(f"LOADING: {url}")
        response = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = min(len(pdf_reader.pages), 25)
            
            text_parts = []
            for i in range(total_pages):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text and len(page_text.strip()) > 50:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error reading page {i+1}: {e}")
                    continue
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"PDF: {total_pages} pages, {len(full_text):,} chars")
            
            return [Document(page_content=full_text, metadata={"source": url, "type": "pdf", "pages": total_pages})]
        
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            clean_text = '\n'.join(line for line in lines if line)
            
            logger.info(f"HTML: {len(clean_text):,} chars")
            return [Document(page_content=clean_text, metadata={"source": url, "type": "html"})]
        
    except Exception as e:
        logger.error(f"Load error: {e}")
        raise

class BalancedRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing BALANCED RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            # Working 8B model
            self.chat_model = ChatTogether(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=0,
                max_tokens=2500  # More tokens for better answers
            )

            # Balanced chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1600,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", "! ", "? ", " "]
            )

            self.initialized = True
            logger.info("BALANCED RAG engine ready!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def smart_chunk_selection(self, chunks: List[Document]) -> List[Document]:
        """Smart selection for good coverage"""
        total = len(chunks)
        target = min(40, total)  # Max 40 chunks for speed, but good coverage
        
        if total <= target:
            return chunks
        
        # Strategic selection
        start_count = int(target * 0.45)  # 45% from start
        middle_count = int(target * 0.25)  # 25% from middle
        end_count = target - start_count - middle_count  # 30% from end
        
        middle_start = total // 2 - middle_count // 2
        
        selected = (chunks[:start_count] + 
                   chunks[middle_start:middle_start + middle_count] + 
                   chunks[-end_count:])
        
        logger.info(f"SELECTION: {len(selected)}/{total} chunks (Start={start_count}, Middle={middle_count}, End={end_count})")
        return selected

    def _balanced_query(self, vectorstore: BalancedVectorStore, query: str) -> str:
        """Balanced query with good context"""
        docs = vectorstore.similarity_search(query, k=8)
        context = " ".join([doc.page_content for doc in docs])[:2800]  # Good context size
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Better prompt for accuracy
        system_content = """You are an expert insurance policy analyst. Answer questions accurately based on the document.

INSTRUCTIONS:
- Input questions are separated by " | "
- Output answers MUST be separated by " | " in the same order
- Provide specific, detailed answers using information from the document
- Include exact numbers, percentages, time periods, conditions when available
- If information is not found, state "Information not available in the document"
- Be comprehensive but concise

CRITICAL: Separate each answer with " | " and maintain exact question order."""

        human_content = f"""Answer these insurance questions based on the document:

Questions: {query}

Document Context: {context}

Provide detailed answers separated by " | ":"""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_balanced(self, url: str, questions: List[str]) -> List[str]:
        """Balanced processing - good accuracy under 30 seconds"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=28.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="28 second timeout exceeded")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        total_start = time.time()
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED!")
        else:
            # Document processing
            docs = balanced_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            # Smart chunk selection
            selected_chunks = self.smart_chunk_selection(chunks)
            
            # Balanced embedding
            vectorstore = BalancedVectorStore(self.embeddings)
            vectorstore.add_documents_balanced(selected_chunks)
            
            self.vectorstore_cache[url_hash] = vectorstore
        
        # Query processing
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._balanced_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start
        logger.info(f"BALANCED: Query={query_time:.1f}s, Total={total_time:.1f}s")
        
        # Improved answer parsing
        answers = []
        raw_splits = response.split(" | ")
        
        for split in raw_splits:
            cleaned = split.strip()
            if (cleaned and len(cleaned) > 10 and 
                not cleaned.lower().startswith(('question:', 'answer:', 'q:', 'a:'))):
                answers.append(cleaned)
        
        # Ensure correct count
        while len(answers) < len(questions):
            answers.append("Information not available in the document.")
        
        logger.info(f"PARSED: {len(answers)} answers for {len(questions)} questions")
        return answers[:len(questions)]

# Global engine
rag_engine = BalancedRAGEngine()

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
        logger.info("BALANCED RAG ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="BALANCED RAG API", version="7.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Balanced processing - good accuracy under 30 seconds"""
    try:
        logger.info(f"BALANCED: {len(request.questions)} questions")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_balanced(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"BALANCED: Completed in {total_time:.1f}s")
        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": "balanced_accuracy_speed",
        "max_chunks": 35,
        "model": "Meta-Llama-3.1-8B-Instruct-Turbo",
        "target_time": "<28_seconds"
    }

@app.get("/")
async def root():
    return {"message": "BALANCED RAG API - Good Accuracy Under 30 Seconds"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
