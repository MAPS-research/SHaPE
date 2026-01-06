"""
OpenAI API Client

Supports calling various OpenAI models including GPT-4, GPT-4o, etc.
"""

import os
from typing import Dict, Any, List
from openai import OpenAI
from .config import load_env_manually, get_env_path


class OpenAIAnswerer:
    """OpenAI API client for answering questions."""
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = "gpt-4o-mini",
                 temperature: float = 1.0, 
                 max_completion_tokens: int = 8000):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, loads from .env or environment.
            model: Model name to use
            temperature: Sampling temperature
            max_completion_tokens: Maximum tokens in completion
        """
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        
        # Load API key from .env file or environment
        if not api_key:
            env_config = load_env_manually(get_env_path())
            api_key = env_config.get('OPENAI_API_KEY')
            if not api_key:
                api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
        
        self.client = OpenAI(api_key=api_key)
        
        print(f"OpenAI client initialized")
        print(f"   Model: {model}")
        print(f"   Temperature: {temperature}")
        print(f"   Max Tokens: {max_completion_tokens}")
    
    def answer_question(self, 
                       question: str, 
                       system_prompt: str, 
                       attack_prefix: str = "",
                       context: Dict[str, Any] = None) -> str:
        """
        Answer a question using OpenAI.
        
        Args:
            question: Question text
            system_prompt: System prompt
            attack_prefix: Optional prefix for the user message
            context: Optional context information
            
        Returns:
            Model's answer
        """
        user_prompt = f"""{attack_prefix}:

Question: {question}"""
        
        if context:
            user_prompt += f"\n\nKnowledge context: {context.get('Knowledge_name', '')}\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            print(f"Calling OpenAI model: {self.model}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                print(f"  Token usage: input={usage.prompt_tokens}, output={usage.completion_tokens}, total={usage.total_tokens}")
            
            return answer
            
        except Exception as e:
            error_msg = f"OpenAI API call failed: {str(e)}"
            print(f"  Error: {error_msg}")
            return error_msg
    
    def chat_messages(self, messages: List[dict]) -> str:
        """
        Send raw chat messages to the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Model's response text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            print("OpenAI connection test successful")
            return True
        except Exception as e:
            print(f"OpenAI connection test failed: {e}")
            return False


def create_openai_answerer(model: str = None, **kwargs) -> OpenAIAnswerer:
    """
    Convenience function to create OpenAI client.
    
    Args:
        model: Model name or shorthand
        **kwargs: Additional arguments for OpenAIAnswerer
        
    Returns:
        Configured OpenAIAnswerer instance
    """
    model_configs = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-3.5-turbo": "gpt-3.5-turbo"
    }
    
    if model and model in model_configs:
        model = model_configs[model]
    elif not model:
        model = "gpt-4o-mini"
    
    return OpenAIAnswerer(model=model, **kwargs)


if __name__ == "__main__":
    print("Testing OpenAI client...")
    try:
        answerer = create_openai_answerer("gpt-4o-mini")
        if answerer.test_connection():
            print("OpenAI client test successful")
        else:
            print("OpenAI client test failed")
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Please ensure OPENAI_API_KEY is set in .env file")

