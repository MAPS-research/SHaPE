"""
API Clients Module

Provides unified interfaces to various LLM APIs including:
- OpenAI (GPT models)
- Google Gemini
- Anthropic Claude
- Together AI (open-source models)
- OpenRouter
- EduChat (local deployment)

All clients implement a consistent interface:
- answer_question(question, system_prompt, attack_prefix, context) -> str
- chat_messages(messages: List[dict]) -> str
- test_connection() -> bool
"""

from .openai_client import OpenAIAnswerer, create_openai_answerer
from .gemini_client import GeminiAnswerer, create_gemini_answerer
from .claude_client import ClaudeAnswerer, create_claude_answerer
from .together_ai_client import TogetherAIAnswerer, create_together_ai_answerer
from .openrouter_client import OpenRouterAnswerer
from .educhat_client import EduChatAnswerer, create_educhat_answerer
from .config import load_env_manually

__all__ = [
    'OpenAIAnswerer', 'create_openai_answerer',
    'GeminiAnswerer', 'create_gemini_answerer',
    'ClaudeAnswerer', 'create_claude_answerer',
    'TogetherAIAnswerer', 'create_together_ai_answerer',
    'OpenRouterAnswerer',
    'EduChatAnswerer', 'create_educhat_answerer',
    'load_env_manually',
]

