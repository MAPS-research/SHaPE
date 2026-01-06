"""
OpenRouter API Client (OpenAI-compatible)

Supports calling various models through OpenRouter.
"""

import os
from typing import Dict, Any, List, Tuple
from openai import OpenAI
from .config import load_env_manually, get_env_path


class OpenRouterAnswerer:
    """OpenRouter API client for answering questions."""

    def __init__(self,
                 api_key: str = None,
                 model: str = "qwen/qwen-7b-chat",
                 temperature: float = 1.0,
                 max_tokens: int = 8000,
                 base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize OpenRouter client.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

        if not api_key:
            env_config = load_env_manually(get_env_path())
            api_key = env_config.get('OPENROUTER_API_KEY')

            env_max_tokens = env_config.get('OPENROUTER_MAX_TOKENS')
            if env_max_tokens:
                try:
                    self.max_tokens = max(self.max_tokens, int(env_max_tokens))
                except ValueError:
                    pass

            if not api_key:
                api_key = os.environ.get('OPENROUTER_API_KEY')

        if not api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        print("OpenRouter client initialized")
        print(f"   Model: {model}")
        print(f"   Temperature: {temperature}")
        print(f"   Max Tokens: {self.max_tokens}")
        print(f"   API URL: {base_url}")

    def _extract_response_text(self, response) -> Tuple[str, str]:
        """Safely extract response text."""
        choices = getattr(response, "choices", None) or []
        for choice in choices:
            message = getattr(choice, "message", None)
            text = getattr(message, "content", None) if message else None
            if text:
                return text.strip(), ""
        return "", "no valid choices"

    def answer_question(self,
                        question: str,
                        system_prompt: str,
                        attack_prefix: str = "",
                        context: Dict[str, Any] = None) -> str:
        """Answer a question using OpenRouter."""
        user_prompt = f"""{attack_prefix}:

Question: {question}"""

        if context:
            user_prompt += f"\n\nKnowledge context: {context.get('Knowledge_name', '')}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            print(f"Calling OpenRouter model: {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer, detail = self._extract_response_text(response)
            if not answer:
                error_msg = f"OpenRouter returned empty response: {detail}"
                print(f"  Error: {error_msg}")
                return error_msg

            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                print(f"  Token usage: input={usage.prompt_tokens}, output={usage.completion_tokens}")

            return answer

        except Exception as e:
            error_msg = f"OpenRouter API call failed: {str(e)}"
            print(f"  Error: {error_msg}")
            return error_msg

    def chat_messages(self, messages: List[dict]) -> str:
        """Send raw chat messages to the model."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        text, detail = self._extract_response_text(response)
        if not text:
            print(f"  OpenRouter returned empty response: {detail}")
            return f"OpenRouterEmptyResponse: {detail}"
        return text

    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            text, _ = self._extract_response_text(response)
            if text:
                print("OpenRouter connection test successful")
                return True
            return False
        except Exception as e:
            print(f"OpenRouter connection test failed: {e}")
            return False

