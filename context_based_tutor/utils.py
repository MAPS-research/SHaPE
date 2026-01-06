"""
Utility functions for the Context-Based Tutor module.
Handles knowledge graph loading, student state parsing, and helper functions.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def load_knowledge_graph(adjacency_csv_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load the knowledge graph adjacency matrix from CSV.
    
    The CSV format:
    - First row: column names (knowledge points)
    - First column: row indices (knowledge points), must match column names and order
    
    Args:
        adjacency_csv_path: Path to the adjacency matrix CSV file
        
    Returns:
        Tuple of (knowledge point names list, adjacency matrix as int array)
        
    Raises:
        ValueError: If row and column labels don't match
    """
    df = pd.read_csv(adjacency_csv_path, header=0, index_col=0)
    
    # Column names are the knowledge point vocabulary
    topics = df.columns.tolist()
    
    # Adjacency matrix (square matrix)
    adj = df.values.astype(int)
    
    # Safety check: row names must match column names exactly
    if list(df.index) != topics:
        raise ValueError("Row and column labels do not match. Please check the CSV file.")
    
    return topics, adj


def load_student_mastery(student_csv_path: str, topics_vocab: List[str]) -> Tuple[List[str], List[str]]:
    """
    Load student knowledge state matrix.
    
    The diagonal elements (1/0) indicate whether each knowledge point is mastered/unmastered.
    Row/column order and naming must exactly match the knowledge graph.
    
    Args:
        student_csv_path: Path to student state CSV file
        topics_vocab: Knowledge point vocabulary (must match exactly)
        
    Returns:
        Tuple of (mastered KP list, unmastered KP list)
        
    Raises:
        ValueError: If matrix is not square or doesn't match knowledge graph
    """
    df = pd.read_csv(student_csv_path, header=0, index_col=0)
    
    if df.shape[0] != df.shape[1]:
        raise ValueError("Student matrix is not square.")
    
    # Ensure consistency with knowledge graph
    if list(df.index) != topics_vocab or list(df.columns) != topics_vocab:
        raise ValueError("Student state row/columns do not match knowledge graph. Please unify naming and order.")
    
    mat = df.values.astype(int)
    diag = np.diag(mat)
    
    mastered = [topics_vocab[i] for i, v in enumerate(diag) if v == 1]
    unmastered = [topics_vocab[i] for i, v in enumerate(diag) if v == 0]
    
    return mastered, unmastered


def dedup_keep_order(lst: List[str]) -> List[str]:
    """
    Deduplicate a list while preserving first-occurrence order.
    
    Args:
        lst: Input list with possible duplicates
        
    Returns:
        Deduplicated list maintaining original order
    """
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

