"""
Together AI API Client

Supports calling various open-source models through Together AI.
"""

import os
from typing import Dict, Any, List, Tuple
from openai import OpenAI
from .config import load_env_manually, get_env_path


class TogetherAIAnswerer:
    """Together AI client for answering questions."""
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = "meta-llama/Llama-2-7b-chat-hf",
                 temperature: float = 1.0, 
                 max_tokens: int = 8000,
                 base_url: str = "https://api.together.xyz/v1"):
        """
        Initialize Together AI client.
        
        Args:
            api_key: Together AI API key
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            base_url: API base URL
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        # Load API key
        if not api_key:
            env_config = load_env_manually(get_env_path())
            api_key = env_config.get('TOGETHER_API_KEY')

            env_max_tokens = env_config.get('TOGETHER_MAX_TOKENS')
            if env_max_tokens:
                try:
                    self.max_tokens = max(self.max_tokens, int(env_max_tokens))
                except ValueError:
                    pass
            
            if not api_key:
                api_key = os.environ.get('TOGETHER_API_KEY')
        
        if not api_key:
            raise ValueError("Together AI API key not found. Please set TOGETHER_API_KEY in .env file.")
        
        # Initialize OpenAI-compatible client
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        print(f"Together AI client initialized")
        print(f"   Model: {model}")
        print(f"   Temperature: {temperature}")
        print(f"   Max Tokens: {self.max_tokens}")
        print(f"   API URL: {base_url}")

    def _extract_response_text(self, response) -> Tuple[str, str]:
        """Safely extract response text."""
        choices = getattr(response, "choices", None) or []
        for idx, choice in enumerate(choices):
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
        """Answer a question using Together AI."""
        user_prompt = f"""{attack_prefix}:

Question: {question}"""
        
        if context:
            user_prompt += f"\n\nKnowledge context: {context.get('Knowledge_name', '')}\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            print(f"Calling Together AI model: {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=60.0
            )

            answer, detail = self._extract_response_text(response)
            if not answer:
                error_msg = f"Together AI returned empty response: {detail}"
                print(f"  Error: {error_msg}")
                return error_msg
            
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                print(f"  Token usage: input={usage.prompt_tokens}, output={usage.completion_tokens}")
            
            return answer
            
        except Exception as e:
            error_msg = f"Together AI API call failed: {str(e)}"
            print(f"  Error: {error_msg}")
            return error_msg
    
    def chat_messages(self, messages: List[dict]) -> str:
        """Send raw chat messages to the model."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=60.0
        )
        text, detail = self._extract_response_text(response)
        if not text:
            print(f"  Together AI returned empty response: {detail}")
            return f"TogetherEmptyResponse: {detail}"
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
                print("Together AI connection test successful")
                return True
            print("Together AI connection test failed: empty response")
            return False
        except Exception as e:
            print(f"Together AI connection test failed: {e}")
            return False


def create_together_ai_answerer(model: str = None, **kwargs) -> TogetherAIAnswerer:
    """Convenience function to create Together AI client."""
    model_configs = {
        "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
        "llama-2-13b": "meta-llama/Llama-2-13b-chat-hf",
        "llama-2-70b": "meta-llama/Llama-2-70b-chat-hf",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }
    
    if model and model in model_configs:
        model = model_configs[model]
    elif not model:
        model = "meta-llama/Llama-2-7b-chat-hf"
    
    return TogetherAIAnswerer(model=model, **kwargs)


if __name__ == "__main__":
    print("Testing Together AI client...")
    try:
        answerer = create_together_ai_answerer("llama-2-7b")
        if answerer.test_connection():
            print("Together AI client test successful")
        else:
            print("Together AI client test failed")
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Please ensure TOGETHER_API_KEY is set in .env file")

