import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.schema import BaseMessage

load_dotenv()

class LLMService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.models = self._initialize_models()

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize LangChain models"""
        models = {}
        
        if self.openai_api_key:
            models.update({
                'GPT 4o - Mini': ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=self.openai_api_key,
                    temperature=0.75
                ),
                'GPT 4o': ChatOpenAI(
                    model="gpt-4o",
                    api_key=self.openai_api_key,
                    temperature=0.75
                ),
                'GPT 3.5 Turbo': ChatOpenAI(
                    model="gpt-3.5-turbo",
                    api_key=self.openai_api_key,
                    temperature=0.75
                )
            })
        
        if self.gemini_api_key:
            models.update({
                'Gemini Pro': ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    google_api_key=self.gemini_api_key,
                    temperature=0.75
                ),
                'Gemini Flash': ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=self.gemini_api_key,
                    temperature=0.75
                )
            })
        
        return models

    async def generate_response(self, user_query: str, context: str, config: Dict[str, Any]) -> str:
        """Generate LLM response using LangChain"""
        try:
            model_name = config.get('model', 'GPT 4o - Mini')
            prompt_prefix = config.get('prompt', 'You are a helpful assistant.')
            temperature = config.get('temperature', 0.75)
            api_key = config.get('apiKey', None)

            # Get or create model instance
            llm = self._get_model_instance(model_name, temperature, api_key)
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_prefix),
                ("human", "Context: {context}\n\nUser Query: {query}")
            ])
            
            # Create chain
            chain = prompt | llm | StrOutputParser()
            
            # Generate response
            response = await chain.ainvoke({
                "context": context,
                "query": user_query
            })
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _get_model_instance(self, model_name: str, temperature: float, api_key: Optional[str] = None):
        """Get or create model instance with custom parameters"""
        if model_name in self.models:
            # Clone model with new temperature
            base_model = self.models[model_name]
            if hasattr(base_model, 'model_name') and 'gpt' in base_model.model_name:
                return ChatOpenAI(
                    model=base_model.model_name,
                    api_key=api_key or self.openai_api_key,
                    temperature=temperature
                )
            elif hasattr(base_model, 'model') and 'gemini' in base_model.model:
                return ChatGoogleGenerativeAI(
                    model=base_model.model,
                    google_api_key=api_key or self.gemini_api_key,
                    temperature=temperature
                )
        
        # Check model name to determine correct provider for fallback
        if 'gemini' in model_name.lower() or 'flash' in model_name.lower():
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key or self.gemini_api_key,
                temperature=temperature
            )
        
        # Default fallback to OpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key or self.openai_api_key,
            temperature=temperature
        )

    async def create_chain(self, prompt_template: str, model_name: str = 'GPT 4o - Mini', **kwargs) -> LLMChain:
        """Create a LangChain LLMChain for reusable workflows"""
        llm = self._get_model_instance(model_name, kwargs.get('temperature', 0.75))
        prompt = PromptTemplate.from_template(prompt_template)
        return LLMChain(llm=llm, prompt=prompt)

    async def batch_generate(self, queries: list, context: str, config: Dict[str, Any]) -> list:
        """Generate responses for multiple queries in batch"""
        try:
            model_name = config.get('model', 'GPT 4o - Mini')
            llm = self._get_model_instance(model_name, config.get('temperature', 0.75))
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", config.get('prompt', 'You are a helpful assistant.')),
                ("human", "Context: {context}\n\nUser Query: {query}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            
            # Process in batch
            inputs = [{"context": context, "query": query} for query in queries]
            responses = await chain.abatch(inputs)
            
            return responses
            
        except Exception as e:
            return [f"Error generating response: {str(e)}" for _ in queries]
