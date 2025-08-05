import os
import time
import logging
import hashlib
import numpy as np
import asyncio
import requests
import io
import PyPDF2
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

# Configuration
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CHAT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MAX_CHUNKS = 25
BATCH_SIZE = 32

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

class OptimizedVectorStore:
    def __init__(self):
        self.embeddings = TogetherEmbeddings(model=EMBEDDING_MODEL)
        self.documents = []
        self.vectors = []
    
    def add_documents(self, documents: List[Document]):
        start_time = time.time()
        valid_docs = [doc for doc in documents if doc.page_content.strip()]
        texts = [doc.page_content for doc in valid_docs]
        vectors = []
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            for attempt in range(3):
                try:
                    vectors.extend(self.embeddings.embed_documents(batch))
                    time.sleep(0.1)
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"Failed batch {i//BATCH_SIZE}: {str(e)[:50]}")
                        vectors.extend([None] * len(batch))
                    time.sleep(0.5 * (attempt + 1))
        
        success_count = 0
        for doc, vector in zip(valid_docs, vectors):
            if vector is not None:
                self.documents.append(doc)
                self.vectors.append(vector)
                success_count += 1
        
        logger.info(f"Embedded {success_count}/{len(documents)} chunks in {time.time()-start_time:.1f}s")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            vectors_array = np.array(self.vectors)
            query_norm = np.linalg.norm(query_vector)
            doc_norms = np.linalg.norm(vectors_array, axis=1)
            
            similarities = np.dot(vectors_array, query_vector) / (doc_norms * query_norm)
            top_indices = np.argsort(similarities)[-k:][::-1]
            return [self.documents[i] for i in top_indices]
        except Exception:
            return self.documents[:k]

class OptimizedRAGEngine:
    def __init__(self):
        self.vectorstore_cache = {}
        self.initialized = False
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". "]
        )
        self.chat_model = None
    
    def initialize(self):
        if not self.initialized:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "your-api-key")
            self.chat_model = ChatTogether(
                model=CHAT_MODEL,
                temperature=0.2,
                max_tokens=1500,
                together_api_key=os.environ["TOGETHER_API_KEY"]  # Explicitly pass API key
            )
            self.initialized = True
            logger.info("RAG engine initialized")

    async def process(self, url: str, questions: List[str]) -> List[str]:
        self.initialize()
        try:
            return await asyncio.wait_for(
                self._process(url, questions),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            return ["Info not found - timeout"] * len(questions)

    async def _process(self, url: str, questions: List[str]) -> List[str]:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        vectorstore = self.vectorstore_cache.get(url_hash)
        
        if not vectorstore:
            text = self._load_document(url)
            docs = [Document(page_content=text)]
            chunks = self.text_splitter.split_documents(docs)
            
            if len(chunks) > MAX_CHUNKS:
                chunks = chunks[:10] + chunks[len(chunks)//2-5:len(chunks)//2+5] + chunks[-5:]
            
            vectorstore = OptimizedVectorStore()
            vectorstore.add_documents(chunks)
            self.vectorstore_cache[url_hash] = vectorstore
        
        return await self._generate_answers(vectorstore, questions)

    def _load_document(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            if 'pdf' in response.headers.get('content-type', '').lower():
                pdf_file = io.BytesIO(response.content)
                reader = PyPDF2.PdfReader(pdf_file)
                return "\n\n".join(page.extract_text() for page in reader.pages)
            return response.text[:100000]
        except Exception as e:
            logger.error(f"Document load error: {e}")
            raise HTTPException(status_code=400, detail="Document loading failed")

    async def _generate_answers(self, vectorstore: OptimizedVectorStore, questions: List[str]) -> List[str]:
        context = ""
        for q in questions[:3]:
            docs = vectorstore.similarity_search(q, k=3)
            context += " ".join(d.page_content for d in docs)[:2000] + "\n\n"
        
        prompt = f"""Answer these questions based on the context:
Context: {context}

Questions: {" | ".join(questions)}

Format requirements:
- Answer in order separated by " | "
- Be concise (1-2 sentences)
- If unsure: "Info not found\""""

        try:
            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            answers = response.content.split(" | ")
            return [a.strip() for a in answers][:len(questions)]
        except Exception:
            return ["Info not found"] * len(questions)

# Initialize engine
rag_engine = OptimizedRAGEngine()

# FastAPI app
app = FastAPI(title="Optimized RAG API")

@app.on_event("startup")
async def startup_event():
    rag_engine.initialize()

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(lambda: verify_token(EXPECTED_TOKEN))
):
    try:
        logger.info(f"Processing {len(request.questions)} questions")
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
        "max_chunks": MAX_CHUNKS,
        "model": CHAT_MODEL
    }

def verify_token(expected: str, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    token = authorization.split("Bearer ")[-1]
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=30)
