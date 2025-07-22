import os
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    TextLoader, 
    CSVLoader,
    JSONLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    PythonCodeTextSplitter
)
from langchain.schema import Document
from fastapi import UploadFile
import tempfile
from .knowledge_base_service import KnowledgeBaseService

class FileProcessor:
    def __init__(self):
        self.kb_service = KnowledgeBaseService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.token_splitter = TokenTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.code_splitter = PythonCodeTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    async def process_pdf(self, file: UploadFile) -> List[Document]:
        """Process PDF file using LangChain loaders"""
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Use LangChain PDF loader
            loader = PyMuPDFLoader(temp_file_path)
            documents = loader.load()
            
            # Split documents into chunks
            split_documents = self.text_splitter.split_documents(documents)
            
            return split_documents
        
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    async def load_document(self, file_path: str) -> List[Document]:
        """Load document using appropriate LangChain loader"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                loader = PyMuPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
            elif file_extension == '.json':
                loader = JSONLoader(file_path, jq_schema='.', text_content=False)
            elif file_extension in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension in ['.ppt', '.pptx']:
                loader = UnstructuredPowerPointLoader(file_path)
            else:
                # Default to text loader
                loader = TextLoader(file_path, encoding='utf-8')
            
            documents = loader.load()
            return documents
            
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            return []
    
    async def split_documents(self, documents: List[Document], splitter_type: str = "recursive") -> List[Document]:
        """Split documents using specified splitter"""
        try:
            if splitter_type == "token":
                splitter = self.token_splitter
            elif splitter_type == "code":
                splitter = self.code_splitter
            else:
                splitter = self.text_splitter
            
            split_docs = splitter.split_documents(documents)
            return split_docs
            
        except Exception as e:
            print(f"Error splitting documents: {e}")
            return documents

    async def process_file(self, file_path: str, splitter_type: str = "recursive") -> List[Document]:
        """Process file and return split documents"""
        try:
            # Load documents
            documents = await self.load_document(file_path)
            
            if not documents:
                return []
            
            # Split documents
            split_documents = await self.split_documents(documents, splitter_type)
            
            return split_documents
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []

    async def process_multiple_files(self, file_paths: List[str], splitter_type: str = "recursive") -> List[Document]:
        """Process multiple files and return combined documents"""
        all_documents = []
        
        for file_path in file_paths:
            documents = await self.process_file(file_path, splitter_type)
            all_documents.extend(documents)
        
        return all_documents

    async def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file"""
        try:
            stat = os.stat(file_path)
            metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": stat.st_size,
                "file_extension": os.path.splitext(file_path)[1],
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime
            }
            return metadata
            
        except Exception as e:
            print(f"Error extracting metadata from {file_path}: {e}")
            return {}

    async def create_document_summary(self, documents: List[Document]) -> str:
        """Create a summary of processed documents"""
        try:
            total_docs = len(documents)
            total_chars = sum(len(doc.page_content) for doc in documents)
            
            file_types = {}
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                ext = os.path.splitext(source)[1] or 'unknown'
                file_types[ext] = file_types.get(ext, 0) + 1
            
            summary = f"""
            Document Processing Summary:
            - Total documents: {total_docs}
            - Total characters: {total_chars:,}
            - File types: {', '.join([f'{ext}: {count}' for ext, count in file_types.items()])}
            """
            
            return summary.strip()
            
        except Exception as e:
            print(f"Error creating document summary: {e}")
            return "Error creating summary"

    async def store_embeddings_from_documents(self, stack_id: str, documents: List[Document], api_key: str, embedding_model: str) -> bool:
        """Store embeddings from document objects"""
        try:
            # Extract text chunks from documents
            chunks = [doc.page_content for doc in documents]
            return await self.kb_service.store_embeddings(stack_id, chunks, api_key, embedding_model)
        except Exception as e:
            print(f"Error storing embeddings from documents: {e}")
            return False