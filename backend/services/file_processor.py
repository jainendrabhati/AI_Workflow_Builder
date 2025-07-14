import fitz  # PyMuPDF
import os
from typing import List
from fastapi import UploadFile
import tempfile
from .knowledge_base_service import KnowledgeBaseService

class FileProcessor:
    def __init__(self):
        self.kb_service = KnowledgeBaseService()
    
    async def process_pdf(self, file: UploadFile) -> List[str]:
        """Process PDF file and extract text chunks"""
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract text from PDF
            text = self._extract_text_from_pdf(temp_file_path)
            
            # Split into chunks
            chunks = self._split_text_into_chunks(text)
            
            return chunks
        
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        
        doc = fitz.open(file_path)
        text = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        
        doc.close()
        return text
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    async def store_embeddings(self, stack_id: str, chunks: List[str]) -> bool:
        """Store embeddings for processed chunks"""
        return await self.kb_service.store_embeddings(stack_id, chunks)