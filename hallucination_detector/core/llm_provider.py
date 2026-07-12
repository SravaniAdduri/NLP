"""
LLM Provider Module
Provides access to language models for text generation.
Supports:
- HuggingFace Inference API (FREE, cloud-based, no GPU needed)
- Ollama (local server)
- HuggingFace Transformers (local, requires GPU/RAM)
All models are free and open-source.
"""

import os
from typing import Optional, List
from abc import ABC, abstractmethod

from loguru import logger


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError


class HuggingFaceInferenceProvider(BaseLLMProvider):
    """
    FREE cloud-based LLM using HuggingFace Inference API.
    No local GPU needed. Just requires a free HF token.
    
    Supported models (free tier):
    - mistralai/Mistral-7B-Instruct-v0.3
    - HuggingFaceH4/zephyr-7b-beta
    - microsoft/Phi-3-mini-4k-instruct
    - google/gemma-2-2b-it
    """

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", hf_token: str = None):
        """
        Initialize the HuggingFace Inference API provider.
        
        Args:
            model_name: Model to use via the Inference API.
            hf_token: HuggingFace API token. Reads from HF_TOKEN env var if not provided.
        """
        from huggingface_hub import InferenceClient

        self.token = hf_token or os.getenv("HF_TOKEN", "")
        self.model_name = model_name

        if not self.token:
            logger.warning("No HF_TOKEN set. Get a free token at https://huggingface.co/settings/tokens")

        self.client = InferenceClient(model=model_name, token=self.token if self.token else None)
        logger.info(f"HuggingFace Inference API initialized with model: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """
        Generate text using HuggingFace Inference API (free).
        
        Args:
            prompt: Input prompt text.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text string.
        """
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                return_full_text=False,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"HF Inference API error: {e}")
            # Try chat completion as fallback
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                )
                return response.choices[0].message.content.strip()
            except Exception as e2:
                logger.error(f"HF Chat API also failed: {e2}")
                return f"[Error: Could not generate response. Check HF_TOKEN. Details: {str(e)[:100]}]"


class HuggingFaceLLMProvider(BaseLLMProvider):
    """
    Local LLM using HuggingFace Transformers pipeline.
    Loads models like Phi-3-mini, Mistral, Gemma locally.
    Requires significant RAM/GPU.
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
    Priority order:
    1. HuggingFace Inference API (free, cloud, no GPU needed)
    2. Ollama (local server)
    3. HuggingFace Transformers (local, needs GPU/RAM)
    """

    def __init__(
        self,
        provider: str = "huggingface_inference",
        model_name: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the LLM provider.
        
        Args:
            provider: 'huggingface_inference' (free cloud), 'ollama', or 'huggingface_local'.
            model_name: Model name (defaults based on provider).
            ollama_base_url: Ollama server URL.
            hf_token: HuggingFace API token for Inference API.
        """
        self.provider_name = provider

        if provider == "huggingface_inference":
            model = model_name or "mistralai/Mistral-7B-Instruct-v0.3"
            try:
                self._provider = HuggingFaceInferenceProvider(model_name=model, hf_token=hf_token)
                logger.info(f"Using HuggingFace Inference API with model: {model}")
            except Exception as e:
                logger.warning(f"HF Inference API failed: {e}, trying Ollama...")
                self._init_ollama_or_fallback(model_name, ollama_base_url)
        elif provider == "ollama":
            self._init_ollama_or_fallback(model_name, ollama_base_url)
        else:
            self._provider = self._create_local_fallback(model_name)

    def _init_ollama_or_fallback(self, model_name: Optional[str], base_url: str) -> None:
        """Try Ollama, fall back to HF Inference API."""
        model = model_name or "mistral"
        try:
            self._provider = OllamaLLMProvider(model_name=model, base_url=base_url)
            self.provider_name = "ollama"
            logger.info(f"Using Ollama provider with model: {model}")
        except Exception as e:
            logger.warning(f"Ollama unavailable: {e}, using HF Inference API")
            self._provider = HuggingFaceInferenceProvider()
            self.provider_name = "huggingface_inference"

    def _create_local_fallback(self, model_name: Optional[str] = None) -> "HuggingFaceLLMProvider":
        """Create a local HuggingFace LLM provider (requires GPU/RAM)."""
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
            query: The user question.
            context: Retrieved evidence/context.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated response grounded in the context.
        """
        prompt = (
            "You are a helpful and accurate assistant. Answer the question based ONLY on the provided context.\n"
            "If the context does not contain enough information to answer, say so. Do not make up information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer (cite the relevant parts of the context):"
        )

        return self.generate(prompt, max_tokens=max_tokens, temperature=0.2)
