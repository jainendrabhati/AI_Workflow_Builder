import os
import openai
from typing import Dict, Any
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

class LLMService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

    async def generate_response(self, user_query: str, context: str, config: Dict[str, Any]) -> str:
        """Generate LLM response based on query and context"""

        model = config.get('model', 'GPT 4o - Mini')
        prompt_prefix = config.get('prompt', 'You are a helpful assistant.')
        temperature = config.get('temperature', 0.75)
        api_key = config.get('apiKey', None)

        # Compose full prompt
        full_prompt = f"{prompt_prefix}\n\nContext: {context}\n\nUser Query: {user_query}"

        try:
            if 'GPT' in model or 'OpenAI' in model:
                return await self._call_openai(full_prompt, model, temperature, api_key or self.openai_api_key)
            elif 'Gemini' in model or 'gemini' in model:
                return await self._call_gemini(user_query, context, prompt_prefix, temperature, model, api_key or self.gemini_api_key)
            else:
                return await self._call_openai(full_prompt, 'gpt-4o-mini', temperature, api_key or self.openai_api_key)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def _call_openai(self, prompt: str, model: str, temperature: float, api_key: str) -> str:
        """Call OpenAI API"""

        model_mapping = {
            'GPT 4o - Mini': 'gpt-4o-mini',
            'GPT 4o': 'gpt-4o',
            'GPT 3.5 Turbo': 'gpt-3.5-turbo'
        }

        api_model = model_mapping.get(model, 'gpt-4o-mini')

        client = openai.OpenAI(api_key=api_key)

        response = await client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000
        )

        return response.choices[0].message.content

    async def _call_gemini(self, user_query: str, context: str, prompt_prefix: str, temperature: float, model: str, api_key: str) -> str:
        """Call Gemini API"""

        try:
            model_name = 'gemini-1.5-pro' if 'pro' in model.lower() else 'gemini-1.5-flash'
            gen_model = genai.GenerativeModel(model_name)

            prompt = f"{prompt_prefix}\n\nContext: {context}\n\nUser Query: {user_query}"

            response = gen_model.generate_content(
                contents=[{"role": "user", "parts": [prompt]}],
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 1000,
                }
            )

            return response.text.strip()

        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"
