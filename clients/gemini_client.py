"""
Google Gemini API Client

Supports calling Google Gemini models.
"""

import os
from typing import Dict, Any, List, Tuple
import google.generativeai as genai
from .config import load_env_manually, get_env_path


class GeminiAnswerer:
    """Gemini API client for answering questions."""
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = "gemini-1.5-pro",
                 temperature: float = 1.0, 
                 max_tokens: int = 8000):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key. If None, loads from .env or environment.
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
            api_key = env_config.get('GEMINI_API_KEY')
            
            env_max_tokens = env_config.get('GEMINI_MAX_TOKENS')
            if env_max_tokens:
                try:
                    self.max_tokens = max(self.max_tokens, int(env_max_tokens))
                except ValueError:
                    pass
            
            if not api_key:
                api_key = os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in .env file.")
        
        genai.configure(api_key=api_key)
        
        self.generation_config = {
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": self.max_tokens,
        }
        
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        
        self.model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        print(f"Gemini client initialized")
        print(f"   Model: {model}")
        print(f"   Temperature: {temperature}")
        print(f"   Max Tokens: {self.max_tokens}")

    def _extract_response_text(self, response) -> Tuple[str, str]:
        """
        Safely extract response text from Gemini response.
        
        Returns:
            Tuple of (text, detail) where text is empty if no valid content
        """
        try:
            text = response.text
            if text:
                return text.strip(), ""
        except Exception:
            pass

        details = []
        candidates = getattr(response, "candidates", None) or []
        for idx, cand in enumerate(candidates):
            finish = getattr(cand, "finish_reason", None)
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                part_texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
                if part_texts:
                    return "\n".join(part_texts).strip(), ""
            details.append(f"candidate[{idx}].finish_reason={finish}")

        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback:
            details.append(f"prompt_feedback={prompt_feedback}")

        return "", " | ".join(details) if details else "no candidates/parts"
    
    def answer_question(self, 
                       question: str, 
                       system_prompt: str, 
                       attack_prefix: str = "",
                       context: Dict[str, Any] = None) -> str:
        """
        Answer a question using Gemini.
        """
        user_prompt = f"""{attack_prefix}:

Question: {question}"""
        
        if context:
            user_prompt += f"\n\nKnowledge context: {context.get('Knowledge_name', '')}\n"
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            print(f"Calling Gemini model: {self.model}")
            response = self.model_instance.generate_content(full_prompt)

            answer, detail = self._extract_response_text(response)
            if answer:
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    print(f"  Token usage: input={usage.prompt_token_count}, output={usage.candidates_token_count}")
                return answer

            error_msg = f"Gemini returned empty response: {detail}"
            print(f"  Error: {error_msg}")
            return error_msg
                
        except Exception as e:
            error_msg = f"Gemini API call failed: {str(e)}"
            print(f"  Error: {error_msg}")
            return error_msg
    
    def chat_messages(self, messages: List[dict]) -> str:
        """Send raw chat messages to the model."""
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        other = "\n\n".join(m["content"] for m in messages if m.get("role") != "system")
        full_prompt = (system + "\n\n" + other).strip()

        response = self.model_instance.generate_content(full_prompt)
        text, detail = self._extract_response_text(response)
        if not text:
            print(f"  Gemini returned empty response: {detail}")
            return f"GeminiEmptyResponse: {detail}"
        return text
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            response = self.model_instance.generate_content("Hello")
            text, _ = self._extract_response_text(response)
            if text:
                print("Gemini connection test successful")
                return True
            print("Gemini connection test failed: empty response")
            return False
        except Exception as e:
            print(f"Gemini connection test failed: {e}")
            return False


def create_gemini_answerer(model: str = None, **kwargs) -> GeminiAnswerer:
    """Convenience function to create Gemini client."""
    model_configs = {
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-pro": "gemini-pro"
    }
    
    if model and model in model_configs:
        model = model_configs[model]
    elif not model:
        model = "gemini-1.5-pro"
    
    return GeminiAnswerer(model=model, **kwargs)


if __name__ == "__main__":
    print("Testing Gemini client...")
    try:
        answerer = create_gemini_answerer("gemini-1.5-pro")
        if answerer.test_connection():
            print("Gemini client test successful")
        else:
            print("Gemini client test failed")
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Please ensure GEMINI_API_KEY is set in .env file")

