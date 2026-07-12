"""
LLM Provider Module
Provides access to language models for text generation.
Supports: HuggingFace Transformers (local), Ollama (local server).
All models are free and open-source.
"""

from typing import Optional, List
from abc import ABC, abstractmethod

from loguru import logger


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError


class HuggingFaceLLMProvider(BaseLLMProvider):
    """
    Local LLM using HuggingFace Transformers pipeline.
    Loads models like Phi-3-mini, Mistral, Gemma locally.
    """

    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", device: str = None):
        """
        Initialize the HuggingFace LLM provider.
        
        Args:
            model_name: HuggingFace model identifier.
            device: Device for inference. Auto-detected if None.
        """
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch

        logger.info(f"Loading HuggingFace model: {model_name}")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True,
        )

        if device == "cpu":
            self.model = self.model.to(device)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=None if device != "cpu" else -1,
        )
        logger.info(f"Model loaded on device: {device}")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """
        Generate text using the local HuggingFace model.
        
        Args:
            prompt: Input prompt text.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            
        Returns:
            Generated text string.
        """
        messages = [{"role": "user", "content": prompt}]

        # Try chat template first
        try:
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = prompt

        output = self.pipe(
            formatted,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False,
        )

        return output[0]["generated_text"].strip()


class OllamaLLMProvider(BaseLLMProvider):
    """
    LLM provider using Ollama local server.
    Requires Ollama to be installed and running.
    """

    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama LLM provider.
        
        Args:
            model_name: Ollama model name (e.g., 'mistral', 'llama3', 'phi3', 'gemma').
            base_url: Ollama server URL.
        """
        import requests

        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

        # Verify Ollama is running
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available_models = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"Ollama connected. Available models: {available_models}")
        except Exception as e:
            logger.warning(f"Ollama not available at {base_url}: {e}. Will retry on generate.")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt text.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text string.
        """
        import requests

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        response = requests.post(self.api_url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "").strip()


class LLMProvider:
    """
    Factory class that provides the appropriate LLM backend.
    Tries Ollama first, falls back to HuggingFace local model.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model_name: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the LLM provider.
        
        Args:
            provider: 'ollama' or 'huggingface'.
            model_name: Model name (defaults based on provider).
            ollama_base_url: Ollama server URL.
        """
        self.provider_name = provider

        if provider == "ollama":
            model = model_name or "mistral"
            try:
                self._provider = OllamaLLMProvider(model_name=model, base_url=ollama_base_url)
                logger.info(f"Using Ollama provider with model: {model}")
            except Exception as e:
                logger.warning(f"Ollama unavailable, falling back to HuggingFace: {e}")
                self._provider = self._create_huggingface_fallback(model_name)
        else:
            self._provider = self._create_huggingface_fallback(model_name)

    def _create_huggingface_fallback(self, model_name: Optional[str] = None) -> HuggingFaceLLMProvider:
        """Create a HuggingFace LLM provider as fallback."""
        model = model_name or "microsoft/Phi-3-mini-4k-instruct"
        return HuggingFaceLLMProvider(model_name=model)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """
        Generate text using the configured provider.
        
        Args:
            prompt: Input prompt for generation.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text string.
        """
        return self._provider.generate(prompt, max_tokens, temperature)

    def generate_with_context(self, query: str, context: str, max_tokens: int = 512) -> str:
        """
        Generate a response using query and retrieved context.
        
        Args:
            query: User's question.
            context: Retrieved evidence/context.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated response grounded in the context.
        """
        prompt = f"""You are a helpful and accurate assistant. Answer the question based ONLY on the provided context. 
If the context does not contain enough information to answer, say so. Do not make up information.

Context:
{context}

Question: {query}

Answer (cite the relevant parts of the context):"""

        return self.generate(prompt, max_tokens=max_tokens, temperature=0.2)
