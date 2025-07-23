import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

class KnowledgeBaseService:
    def __init__(self):
        self.persist_directory = "./chroma_db_store"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GEMINI_API_KEY")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embedding models"""
        if self.openai_api_key:
            return OpenAIEmbeddings(
                api_key=self.openai_api_key,
                model="text-embedding-3-small"
            )
        elif self.google_api_key:
            return GoogleGenerativeAIEmbeddings(
                google_api_key=self.google_api_key,
                model="models/embedding-001"
            )
        else:
            raise ValueError("No API key provided for embeddings")

    async def store_embeddings(
        self,
        stack_id: str,
        chunks: List[str],
        api_key: str,
        embedding_model: str
    ) -> bool:
        try:
            collection_name = self._sanitize_collection_name(f"stack_{stack_id}")
            print(f"[store_embeddings] Using collection: {collection_name}")

            # Limit chunk size to prevent timeout
            if len(chunks) > 100:
                print(f"[store_embeddings] Warning: Large number of chunks ({len(chunks)}), processing in batches")
                chunks = chunks[:100]  # Limit to first 100 chunks
            
            # Filter out empty chunks and limit content length
            valid_chunks = []
            for chunk in chunks:
                if chunk and chunk.strip():
                    # Limit chunk size to prevent embedding API timeout
                    if len(chunk) > 8000:  # Reasonable limit for embedding models
                        chunk = chunk[:8000] + "..."
                    valid_chunks.append(chunk)
            
            if not valid_chunks:
                print("[store_embeddings] No valid chunks to process")
                return False

            # Create documents from chunks
            documents = [Document(page_content=chunk) for chunk in valid_chunks]
            
            # Get appropriate embeddings with timeout handling
            embeddings = self._get_embeddings_model(api_key, embedding_model)
            
            # Create vector store with timeout protection
            import asyncio
            try:
                vectorstore = await asyncio.wait_for(
                    asyncio.to_thread(
                        Chroma.from_documents,
                        documents=documents,
                        embedding=embeddings,
                        collection_name=collection_name,
                        persist_directory=self.persist_directory,
                        collection_metadata={
                            "stack_id": stack_id,
                            "embedding_model": embedding_model
                        }
                    ),
                    timeout=120  # 2 minute timeout for embedding creation
                )
                
                # Persist the data
                await asyncio.to_thread(vectorstore.persist)
                print("[store_embeddings] Embeddings stored successfully using LangChain.")
                return True
                
            except asyncio.TimeoutError:
                print("[store_embeddings] Timeout error during embedding creation")
                return False

        except Exception as e:
            print(f"[store_embeddings] Error: {e}")
            return False

    async def retrieve_context(
        self,
        stack_id: str,
        query: str,
        config: Dict[str, Any]
    ) -> str:
        try:
            print(f"[retrieve_context] Retrieving context for stack_id: {stack_id}")
            collection_name = self._sanitize_collection_name(f"stack_{stack_id}")
            
            # Get embeddings model with timeout handling
            api_key = config.get("api_key")
            embedding_model = config.get("embedding_model", "text-embedding-3-small")
            embeddings = self._get_embeddings_model(api_key, embedding_model)
            
            # Load existing vector store with timeout protection
            import asyncio
            try:
                vectorstore = await asyncio.wait_for(
                    asyncio.to_thread(
                        Chroma,
                        collection_name=collection_name,
                        embedding_function=embeddings,
                        persist_directory=self.persist_directory
                    ),
                    timeout=30  # 30 second timeout for loading vector store
                )
                
                # Perform similarity search with timeout
                docs = await asyncio.wait_for(
                    asyncio.to_thread(vectorstore.similarity_search, query, 5),
                    timeout=60  # 60 second timeout for similarity search
                )
                
                if docs:
                    context = "\n\n".join([doc.page_content for doc in docs])
                    return context
                else:
                    return "No relevant context found in knowledge base."
                    
            except asyncio.TimeoutError:
                return "Error: Timeout occurred while retrieving context from knowledge base."
            except Exception as vector_error:
                print(f"[retrieve_context] Vector store error: {vector_error}")
                return "Error: Could not access knowledge base. Please ensure embeddings are properly stored."

        except Exception as e:
            print(f"[retrieve_context] Error: {e}")
            return f"Error embedding content: {str(e)}"

    def create_retrieval_qa_chain(self, stack_id: str, llm, config: Dict[str, Any]):
        """Create a RetrievalQA chain for the knowledge base"""
        try:
            collection_name = self._sanitize_collection_name(f"stack_{stack_id}")
            
            # Get embeddings model
            api_key = config.get("api_key")
            embedding_model = config.get("embedding_model", "text-embedding-3-small")
            embeddings = self._get_embeddings_model(api_key, embedding_model)
            
            # Load vector store
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory
            )
            
            # Create retrieval QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True
            )
            
            return qa_chain
            
        except Exception as e:
            print(f"Error creating retrieval QA chain: {e}")
            return None

    def _get_embeddings_model(self, api_key: Optional[str], embedding_model: str):
        """Get the appropriate embeddings model with timeout configuration"""
        try:
            if "openai" in embedding_model.lower() or "text-embedding" in embedding_model:
                return OpenAIEmbeddings(
                    api_key=api_key or self.openai_api_key,
                    model=embedding_model,
                    request_timeout=60,  # 60 second timeout
                    max_retries=3
                )
            elif "gemini" in embedding_model.lower() or "embedding-001" in embedding_model:
                return GoogleGenerativeAIEmbeddings(
                    google_api_key=api_key or self.google_api_key,
                    model="models/embedding-001",
                    request_timeout=60  # 60 second timeout
                )
            else:
                # Default to OpenAI with timeout
                return OpenAIEmbeddings(
                    api_key=api_key or self.openai_api_key,
                    model="text-embedding-3-small",
                    request_timeout=60,
                    max_retries=3
                )
        except Exception as e:
            print(f"Error creating embeddings model: {e}")
            raise

    async def process_documents_from_files(self, file_paths: List[str]) -> List[Document]:
        """Process documents from files using LangChain loaders"""
        documents = []
        
        for file_path in file_paths:
            try:
                if file_path.endswith('.pdf'):
                    loader = PyMuPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                
                docs = loader.load()
                # Split documents into chunks
                split_docs = self.text_splitter.split_documents(docs)
                documents.extend(split_docs)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        return documents

    async def create_knowledge_base_from_files(
        self, 
        stack_id: str, 
        file_paths: List[str], 
        api_key: str, 
        embedding_model: str
    ) -> bool:
        """Create knowledge base directly from files using LangChain"""
        try:
            # Process documents
            documents = await self.process_documents_from_files(file_paths)
            
            if not documents:
                print("No documents processed")
                return False
            
            collection_name = self._sanitize_collection_name(f"stack_{stack_id}")
            embeddings = self._get_embeddings_model(api_key, embedding_model)
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=self.persist_directory,
                collection_metadata={
                    "stack_id": stack_id,
                    "embedding_model": embedding_model
                }
            )
            
            vectorstore.persist()
            print(f"Knowledge base created successfully for stack {stack_id}")
            return True
            
        except Exception as e:
            print(f"Error creating knowledge base from files: {e}")
            return False

    def _sanitize_collection_name(self, name: str) -> str:
        import re
        print("Sanitizing collection name...")
        name = name.replace(" ", "_")
        name = re.sub(r"[^a-zA-Z0-9_\-\.]", "", name)
        return name[:63]
