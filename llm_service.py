import os
import json
import httpx
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables IMMEDIATELY to ensure global instance is correctly configured
load_dotenv()

class LLMService:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.ollama_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/api/chat").strip()
        self.ollama_model = os.getenv("LLM_MODEL", "qwen2.5:7b").strip()

    def call_llm(self, messages: List[Dict], temperature: float = 0.0, max_tokens: int = 500) -> str:
        if self.provider == "openai":
            return self._call_openai(messages, temperature, max_tokens)
        else:
            return self._call_ollama(messages, temperature, max_tokens)

    def _call_openai(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        if not self.openai_api_key:
            return "Error: OPENAI_API_KEY is not set."
        
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.openai_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[LLMService] OpenAI Error: {e}")
            return f"Error calling OpenAI: {e}"

    def _call_ollama(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        try:
            resp = httpx.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=300
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            print(f"[LLMService] Ollama Error: {e}")
            return f"Error calling Ollama: {e}"

    async def stream_llm(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 512):
        """Streaming version of LLM call."""
        if self.provider == "openai":
            async for chunk in self._stream_openai(messages, temperature, max_tokens):
                yield chunk
        else:
            async for chunk in self._stream_ollama(messages, temperature, max_tokens):
                yield chunk

    async def _stream_openai(self, messages: List[Dict], temperature: float, max_tokens: int):
        print(f"[LLMService] Calling OpenAI Stream (model={self.openai_model})...")
        if not self.openai_api_key:
            print("[LLMService] Error: OpenAI API Key missing.")
            yield "Error: OPENAI_API_KEY is not set."
            return
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.openai_model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": True
                    }
                ) as resp:
                    print(f"[LLMService] OpenAI Stream status: {resp.status_code}")
                    resp.raise_for_status()
                    count = 0
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            token = data["choices"][0]["delta"].get("content", "")
                            if token:
                                count += 1
                                yield token
                        except Exception:
                            continue
                    print(f"[LLMService] OpenAI Stream finished. Total chunks: {count}")
        except Exception as e:
            print(f"[LLMService] OpenAI Stream Error: {e}")
            yield f"Error streaming OpenAI: {e}"

    async def _stream_ollama(self, messages: List[Dict], temperature: float, max_tokens: int):
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST",
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    }
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield token
                        except Exception:
                            continue
        except Exception as e:
            print(f"[LLMService] Ollama Stream Error: {e}")
            yield f"Error streaming Ollama: {e}"

# Global instance
llm_service = LLMService()

def call_llm(messages: List[Dict], temperature: float = 0.0, max_tokens: int = 500) -> str:
    return llm_service.call_llm(messages, temperature, max_tokens)

async def stream_llm(messages: List[Dict], temperature: float = 0.1, max_tokens: int = 512):
    async for chunk in llm_service.stream_llm(messages, temperature, max_tokens):
        yield chunk
