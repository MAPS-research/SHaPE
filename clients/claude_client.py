"""
Anthropic Claude API Client

Supports calling Claude models.
"""

import os
from typing import Dict, Any, List, Tuple
import anthropic
from .config import load_env_manually, get_env_path


class ClaudeAnswerer:
    """Claude API client for answering questions."""
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = "claude-3-sonnet-20240229",
                 temperature: float = 1.0, 
                 max_tokens: int = 8000):
        """
        Initialize Claude client.
        
        Args:
            api_key: Claude API key. If None, loads from .env or environment.
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load API key
        if not api_key:
            env_config = load_env_manually(get_env_path())
            api_key = env_config.get('CLAUDE_API_KEY')

            env_max_tokens = env_config.get('CLAUDE_MAX_TOKENS')
            if env_max_tokens:
                try:
                    self.max_tokens = max(self.max_tokens, int(env_max_tokens))
                except ValueError:
                    pass
            
            if not api_key:
                api_key = os.environ.get('CLAUDE_API_KEY')
        
        if not api_key:
            raise ValueError("Claude API key not found. Please set CLAUDE_API_KEY in .env file.")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        print(f"Claude client initialized")
        print(f"   Model: {model}")
        print(f"   Temperature: {temperature}")
        print(f"   Max Tokens: {self.max_tokens}")

    def _extract_response_text(self, response) -> Tuple[str, str]:
        """Safely extract response text from Claude response."""
        parts = getattr(response, "content", None) or []
        texts = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
        if texts:
            return "\n".join(texts).strip(), ""

        stop_reason = getattr(response, "stop_reason", None)
        usage = getattr(response, "usage", None)
        detail = f"stop_reason={stop_reason}"
        if usage:
            detail += f", usage={usage}"
        return "", detail
    
    def answer_question(self, 
                       question: str, 
                       system_prompt: str, 
                       attack_prefix: str = "",
                       context: Dict[str, Any] = None) -> str:
        """Answer a question using Claude."""
        user_prompt = f"""{attack_prefix}:

Question: {question}"""
        
        if context:
            user_prompt += f"\n\nKnowledge context: {context.get('Knowledge_name', '')}\n"
        
        try:
            print(f"Calling Claude model: {self.model}")
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            answer, detail = self._extract_response_text(response)
            if not answer:
                error_msg = f"Claude returned empty response: {detail}"
                print(f"  Error: {error_msg}")
                return error_msg
            
            if hasattr(response, 'usage'):
                usage = response.usage
                print(f"  Token usage: input={usage.input_tokens}, output={usage.output_tokens}")
            
            return answer
            
        except Exception as e:
            error_msg = f"Claude API call failed: {str(e)}"
            print(f"  Error: {error_msg}")
            return error_msg
    
    def chat_messages(self, messages: List[dict]) -> str:
        """Send raw chat messages to the model."""
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system").strip()
        other = "\n\n".join(m["content"] for m in messages if m.get("role") != "system").strip()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": other}],
        )
        text, detail = self._extract_response_text(response)
        if not text:
            print(f"  Claude returned empty response: {detail}")
            return f"ClaudeEmptyResponse: {detail}"
        return text
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            text, _ = self._extract_response_text(response)
            if text:
                print("Claude connection test successful")
                return True
            print("Claude connection test failed: empty response")
            return False
        except Exception as e:
            print(f"Claude connection test failed: {e}")
            return False


def create_claude_answerer(model: str = None, **kwargs) -> ClaudeAnswerer:
    """Convenience function to create Claude client."""
    model_configs = {
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-2.1": "claude-2.1"
    }
    
    if model and model in model_configs:
        model = model_configs[model]
    elif not model:
        model = "claude-3-sonnet-20240229"
    
    return ClaudeAnswerer(model=model, **kwargs)


if __name__ == "__main__":
    print("Testing Claude client...")
    try:
        answerer = create_claude_answerer("claude-3-sonnet")
        if answerer.test_connection():
            print("Claude client test successful")
        else:
            print("Claude client test failed")
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Please ensure CLAUDE_API_KEY is set in .env file")

