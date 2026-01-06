"""
EduChat Local Model Client

Supports calling locally deployed EduChat models via HTTP API.
"""

import os
import requests
from typing import Dict, Any, List
from .config import load_env_manually, get_env_path


class EduChatAnswerer:
    """EduChat local model client for answering questions."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 model: str = "educhat",
                 temperature: float = 1.0, 
                 max_tokens: int = 4000,
                 timeout: int = 60):
        """
        Initialize EduChat client.
        
        Args:
            base_url: Base URL of the EduChat service
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Load configuration from .env
        env_config = load_env_manually(get_env_path())
        if env_config.get('EDUCHAT_BASE_URL'):
            self.base_url = env_config.get('EDUCHAT_BASE_URL')
        
        print(f"EduChat client initialized")
        print(f"   Service URL: {self.base_url}")
        print(f"   Model: {model}")
        print(f"   Temperature: {temperature}")
        print(f"   Max Tokens: {max_tokens}")
        print(f"   Timeout: {timeout}s")
    
    def answer_question(self, 
                       question: str, 
                       system_prompt: str, 
                       attack_prefix: str = "",
                       context: Dict[str, Any] = None) -> str:
        """Answer a question using EduChat."""
        user_prompt = f"""{attack_prefix}:

Question: {question}"""
        
        if context:
            user_prompt += f"\n\nKnowledge context: {context.get('Knowledge_name', '')}\n"
        
        request_data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        try:
            print(f"Calling EduChat model: {self.model}")
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    answer = result["choices"][0]["message"]["content"].strip()
                    
                    if "usage" in result:
                        usage = result["usage"]
                        print(f"  Token usage: input={usage.get('prompt_tokens', 'N/A')}, output={usage.get('completion_tokens', 'N/A')}")
                    
                    return answer
                else:
                    return "EduChat returned invalid format"
            else:
                return f"EduChat API call failed: HTTP {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return f"EduChat request timeout ({self.timeout}s)"
        except requests.exceptions.ConnectionError:
            return f"Cannot connect to EduChat service: {self.base_url}"
        except Exception as e:
            return f"EduChat API call failed: {str(e)}"

    def chat_messages(self, messages: List[dict]) -> str:
        """Send raw chat messages to the model."""
        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise RuntimeError(f"EduChat HTTP {response.status_code}: {response.text}")

        result = response.json()
        return (result.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            # Try health check endpoint
            health_url = f"{self.base_url}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("model_loaded", False):
                    print("EduChat connection test successful, model loaded")
                    return True
                else:
                    print("EduChat service running but model not loaded")
                    return False
        except Exception:
            pass
        
        # Try models endpoint
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                print("EduChat connection test successful")
                return True
        except Exception:
            pass
        
        # Try a simple chat request
        try:
            test_data = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=test_data,
                timeout=10
            )
            if response.status_code == 200:
                print("EduChat connection test successful")
                return True
            elif response.status_code == 503:
                print("EduChat service running but model not loaded")
                return False
        except Exception:
            pass
        
        print("EduChat connection test failed")
        return False


def create_educhat_answerer(base_url: str = None, **kwargs) -> EduChatAnswerer:
    """Convenience function to create EduChat client."""
    if not base_url:
        env_config = load_env_manually(get_env_path())
        base_url = env_config.get('EDUCHAT_BASE_URL', 'http://localhost:8000')
    
    return EduChatAnswerer(base_url=base_url, **kwargs)


if __name__ == "__main__":
    print("Testing EduChat client...")
    try:
        answerer = create_educhat_answerer()
        if answerer.test_connection():
            print("EduChat client test successful")
        else:
            print("EduChat client test failed")
            print("Please ensure EduChat service is running")
    except Exception as e:
        print(f"Initialization failed: {e}")

