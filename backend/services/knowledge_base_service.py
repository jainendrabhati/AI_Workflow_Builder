import os
from typing import List, Dict, Any
import chromadb
import openai
import httpx
from dotenv import load_dotenv

load_dotenv()

class KnowledgeBaseService:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db_store")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GEMINI_API_KEY")

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

            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "stack_id": stack_id,
                    "embedding_model": embedding_model
                }
            )
            print("123456789087654321")
            embeddings = await self._generate_embeddings(
                texts=chunks,
                api_key=api_key,
                embedding_model=embedding_model
            )
            print("Embeddings generated successfully.")
            collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )

            print("[store_embeddings] Embeddings stored successfully.")
            return True

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
            print(f"[retrieve_context] Using collection: {collection_name}")

            # ✅ Safely check if collection exists
            print(f"Checking if collection exists...: {collection_name}")
            available_collections = [c.name for c in self.chroma_client.list_collections()]
            print(f"Available collections: {available_collections}")
            if collection_name not in available_collections:
                return f"[retrieve_context] Error: Collection `{collection_name}` does not exist. Please store embeddings first."

            collection = self.chroma_client.get_collection(name=collection_name)
            embedding_model = "gemini-1.5-flash"
            print("________________________________________________________________________________________")
            print(embedding_model)
            api_key = config.get("api_key")

            query_embedding = await self._generate_embeddings(
                texts=[query],
                api_key=api_key,
                embedding_model=embedding_model
            )

            results = collection.query(
                query_embeddings=query_embedding,
                n_results=5
            )

            if results['documents']:
                context = "\n\n".join(results['documents'][0])
                return context
            else:
                return "No relevant context found in knowledge base."

        except Exception as e:
            return f"[retrieve_context] Error: {e}"

    async def _generate_embeddings(
        self,
        texts: List[str],
        api_key: str = None,
        embedding_model: str = "gemini-1.5-flash"
    ) -> List[List[float]]:
        try:
            if embedding_model == "gemini-1.5-flash":
                print("[_generate_embeddings] Using Gemini for embeddings.")
                return await self._generate_gemini_embeddings(texts)
            else:
                print("[_generate_embeddings] Using OpenAI for embeddings.")
                client = openai.OpenAI(api_key=api_key or self.openai_api_key)
                response = client.embeddings.create(
                    model=embedding_model,
                    input=texts
                )
                return [embedding.embedding for embedding in response.data]

        except Exception as e:
            print(f"[_generate_embeddings] Error: {e}")
            raise

    async def _generate_gemini_embeddings(self, texts: List[str]) -> List[List[float]]:
        GEMINI_EMBEDDING_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"
        headers = {"Content-Type": "application/json"}
        embeddings = []
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(self.chroma_client.list_collections())
        
        async with httpx.AsyncClient() as client:
            for text in texts:
                payload = {
                    "model": "models/embedding-001",
                    "content": {"parts": [{"text": text}]}
                }
                params = {"key": self.google_api_key}
                print(params)
                response = await client.post(GEMINI_EMBEDDING_ENDPOINT, params=params, json=payload, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    embeddings.append(data["embedding"]["values"])
                else:
                    print(f"Gemini embedding error: {response.text}")
                    raise Exception("Failed to generate Gemini embeddings.")
                
        return embeddings

    def _sanitize_collection_name(self, name: str) -> str:
        import re
        print("Sanitizing collection name...")
        name = name.replace(" ", "_")
        name = re.sub(r"[^a-zA-Z0-9_\-\.]", "", name)
        return name[:63]
