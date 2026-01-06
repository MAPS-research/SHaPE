"""
LLM Adapters for the Context-Based Tutor module.
Converts various client types to a unified async interface compatible with LangGraph.
"""

import asyncio
from typing import Any, List, Dict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


def _lc_to_openai_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """
    Convert LangChain message objects to OpenAI-compatible message dicts.
    
    Args:
        messages: List of LangChain BaseMessage objects
        
    Returns:
        List of dicts with 'role' and 'content' keys
    """
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        else:
            # Fallback for AIMessage or custom message types
            role = "assistant"
        out.append({"role": role, "content": getattr(m, "content", str(m))})
    return out


class ClientToAInvokeAdapter:
    """
    Wraps custom API clients into an object that supports:
        await llm.ainvoke([SystemMessage(...), HumanMessage(...)])
    and returns an AIMessage with .content attribute.
    
    Compatible with clients that have either:
    - chat_messages(messages: List[dict]) -> str
    - answer_question(question, system_prompt, attack_prefix, context) -> str
    """

    def __init__(self, client: Any):
        """
        Initialize adapter with a client instance.
        
        Args:
            client: API client with chat_messages() or answer_question() method
            
        Raises:
            TypeError: If client doesn't have required methods
        """
        self.client = client
        if not (hasattr(client, "chat_messages") or hasattr(client, "answer_question")):
            raise TypeError(
                f"Client must have chat_messages() or answer_question(); got {type(client)}"
            )

    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """
        Async invoke the underlying client with LangChain messages.
        
        Args:
            messages: List of LangChain message objects
            **kwargs: Additional arguments (ignored)
            
        Returns:
            AIMessage containing the model's response
        """
        oa_messages = _lc_to_openai_messages(messages)

        # Prefer raw chat_messages to avoid injecting templates
        if hasattr(self.client, "chat_messages"):
            text = await asyncio.to_thread(self.client.chat_messages, oa_messages)
            return AIMessage(content=text or "")

        # Fallback legacy path using answer_question
        system = "\n\n".join(m["content"] for m in oa_messages if m["role"] == "system").strip()
        user = "\n\n".join(m["content"] for m in oa_messages if m["role"] != "system").strip()

        text = await asyncio.to_thread(
            self.client.answer_question,
            user,
            system,
            "",    # attack_prefix
            None,  # context
        )
        return AIMessage(content=text or "")


def adapt_llm(llm: Any) -> Any:
    """
    Adapt various LLM types to a unified interface.
    
    Accepts:
      - None -> returns None
      - LangChain-like LLM with .ainvoke -> returns as-is
      - Custom clients with chat_messages() or answer_question() -> wraps in adapter
    
    Args:
        llm: LLM instance or None
        
    Returns:
        Adapted LLM with .ainvoke() method, or None
        
    Raises:
        TypeError: If llm type is not supported
    """
    if llm is None:
        return None
    if hasattr(llm, "ainvoke"):
        return llm
    if hasattr(llm, "chat_messages") or hasattr(llm, "answer_question"):
        return ClientToAInvokeAdapter(llm)
    raise TypeError(f"llm must have .ainvoke() or be a supported client; got {type(llm)}")

