"""
Main entry point for the Adaptive Tutor module.
Provides interface for running adaptive tutoring with system-prompt-based approach.
"""

import os
import asyncio
from typing import Dict, List, Any

from .prompts import create_adaptive_system_prompt
from .knowledge_graph import KnowledgeGraph


# Default path to adjacency matrix (relative to this file)
DEFAULT_ADJ_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "adjacency_matrix_knowledge_graph.csv")


def run_adaptive_tutor(
    question: str,
    missing_kps: List[str],
    client: Any,
    adjacency_csv_path: str = None,
    attack_prefix: str = "",
    use_dense_state: bool = True
) -> Dict:
    """
    Run the adaptive tutor with system-prompt-based approach.
    
    This approach embeds the student's knowledge state directly in the system prompt
    and sends the question to the LLM for a personalized response.
    
    Args:
        question: The student's question text
        missing_kps: List of knowledge points the student is missing
        client: API client with answer_question() or chat_messages() method
        adjacency_csv_path: Path to knowledge graph CSV. If None, uses default.
        attack_prefix: Optional prefix to add to the user prompt (for jailbreak testing)
        use_dense_state: If True, compute dense state including dependent KPs
        
    Returns:
        Dict containing:
            - answer: The tutor's response
            - system_prompt: The generated system prompt
            - user_prompt: The user prompt sent to the model
            - known_knowledge: List of mastered KPs
            - missing_knowledge: List of missing KPs
    """
    if adjacency_csv_path is None:
        adjacency_csv_path = DEFAULT_ADJ_PATH
    
    # Load knowledge graph
    kg = KnowledgeGraph(adjacency_csv_path)
    
    # Compute knowledge state
    if use_dense_state:
        known_knowledge, missing_knowledge = kg.compute_dense_state(missing_kps)
    else:
        # Simple state: all KPs except missing ones
        missing_knowledge = list(missing_kps)
        known_knowledge = [kp for kp in kg.knowledge_points if kp not in missing_kps]
    
    # Create adaptive system prompt
    system_prompt = create_adaptive_system_prompt(known_knowledge, missing_knowledge)
    
    # Build user prompt
    if attack_prefix and attack_prefix.strip():
        user_prompt = f"{attack_prefix}\n\nQuestion: {question}"
    else:
        user_prompt = f"Directly give me the answer of this question: {question}"
    
    # Call the model
    if hasattr(client, "chat_messages"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        answer = client.chat_messages(messages)
    elif hasattr(client, "answer_question"):
        answer = client.answer_question(
            question=question,
            system_prompt=system_prompt,
            attack_prefix=attack_prefix
        )
    else:
        raise TypeError("Client must have chat_messages() or answer_question() method")
    
    return {
        "answer": answer,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "known_knowledge": known_knowledge,
        "missing_knowledge": missing_knowledge
    }


async def run_adaptive_tutor_async(
    question: str,
    missing_kps: List[str],
    client: Any,
    adjacency_csv_path: str = None,
    attack_prefix: str = "",
    use_dense_state: bool = True
) -> Dict:
    """
    Run the adaptive tutor asynchronously.
    
    Same as run_adaptive_tutor but uses asyncio for non-blocking execution.
    
    Args:
        Same as run_adaptive_tutor
        
    Returns:
        Same as run_adaptive_tutor
    """
    if adjacency_csv_path is None:
        adjacency_csv_path = DEFAULT_ADJ_PATH
    
    # Load knowledge graph
    kg = KnowledgeGraph(adjacency_csv_path)
    
    # Compute knowledge state
    if use_dense_state:
        known_knowledge, missing_knowledge = kg.compute_dense_state(missing_kps)
    else:
        missing_knowledge = list(missing_kps)
        known_knowledge = [kp for kp in kg.knowledge_points if kp not in missing_kps]
    
    # Create adaptive system prompt
    system_prompt = create_adaptive_system_prompt(known_knowledge, missing_knowledge)
    
    # Build user prompt
    if attack_prefix and attack_prefix.strip():
        user_prompt = f"{attack_prefix}\n\nQuestion: {question}"
    else:
        user_prompt = f"Directly give me the answer of this question: {question}"
    
    # Call the model asynchronously
    loop = asyncio.get_event_loop()
    
    if hasattr(client, "chat_messages"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        answer = await loop.run_in_executor(None, client.chat_messages, messages)
    elif hasattr(client, "answer_question"):
        answer = await loop.run_in_executor(
            None,
            lambda: client.answer_question(
                question=question,
                system_prompt=system_prompt,
                attack_prefix=attack_prefix
            )
        )
    else:
        raise TypeError("Client must have chat_messages() or answer_question() method")
    
    return {
        "answer": answer,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "known_knowledge": known_knowledge,
        "missing_knowledge": missing_knowledge
    }


if __name__ == "__main__":
    # Example usage (requires API client to be set up)
    print("Adaptive Tutor Module")
    print("=====================")
    print("Usage:")
    print("  from adaptive_tutor import run_adaptive_tutor")
    print("  from clients import OpenAIAnswerer")
    print("  ")
    print("  client = OpenAIAnswerer(model='gpt-4o-mini')")
    print("  result = run_adaptive_tutor(")
    print("      question='How do I find eigenvalues?',")
    print("      missing_kps=['Eigenvalues and Eigenvectors'],")
    print("      client=client")
    print("  )")
    print("  print(result['answer'])")

