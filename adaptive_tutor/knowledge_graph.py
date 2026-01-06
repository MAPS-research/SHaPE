"""
Knowledge Graph utilities for the Adaptive Tutor module.
Handles loading and processing of knowledge point dependencies.
"""

import csv
from typing import List, Set, Dict, Tuple


class KnowledgeGraph:
    """
    Knowledge Graph class for handling knowledge point dependencies.
    
    The adjacency matrix represents prerequisite relationships:
    - adj_matrix[i][j] = 1 means KP[i] depends on (requires) KP[j]
    """

    def __init__(self, adjacency_csv_path: str):
        """
        Initialize the knowledge graph from an adjacency matrix CSV.
        
        Args:
            adjacency_csv_path: Path to the adjacency matrix CSV file
        """
        self.adjacency_csv_path = adjacency_csv_path
        self.knowledge_points: List[str] = []
        self.adjacency_matrix: List[List[int]] = []
        self.prerequisites: Dict[str, List[str]] = {}
        self._load_adjacency_matrix()

    def _load_adjacency_matrix(self):
        """Load adjacency matrix and build dependency relationships."""
        print("Loading knowledge graph...")

        with open(self.adjacency_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # First row contains knowledge point names (first column is empty)
        self.knowledge_points = rows[0][1:]

        # Build adjacency matrix (starting from second row)
        self.adjacency_matrix = []
        for row in rows[1:]:
            # Skip first column (KP name), take data from second column onwards
            matrix_row = [int(value) for value in row[1:]]
            self.adjacency_matrix.append(matrix_row)

        # Build prerequisite mapping
        self.prerequisites = {}
        for i, kp in enumerate(self.knowledge_points):
            self.prerequisites[kp] = []
            for j, dependency in enumerate(self.adjacency_matrix[i]):
                if dependency == 1:
                    self.prerequisites[kp].append(self.knowledge_points[j])

        print(f"Loaded {len(self.knowledge_points)} knowledge points")

    def get_prerequisites(self, knowledge_point: str) -> List[str]:
        """
        Get direct prerequisites of a knowledge point.
        
        Args:
            knowledge_point: Name of the knowledge point
            
        Returns:
            List of prerequisite knowledge point names
        """
        return self.prerequisites.get(knowledge_point, [])

    def get_all_prerequisites(self, knowledge_points: List[str]) -> Set[str]:
        """
        Get all prerequisites recursively for multiple knowledge points.
        
        Args:
            knowledge_points: List of knowledge point names
            
        Returns:
            Set of all prerequisite knowledge points (including the input KPs)
        """
        all_prereqs = set()
        to_check = set(knowledge_points)

        while to_check:
            current = to_check.pop()
            if current not in all_prereqs:
                prereqs = set(self.get_prerequisites(current))
                all_prereqs.add(current)
                to_check.update(prereqs - all_prereqs)

        return all_prereqs

    def is_valid_missing_combination(self, required_kps: List[str], missing_kps: List[str]) -> bool:
        """
        Check if a combination of missing knowledge points is valid.
        
        A combination is valid if no missing KP is a prerequisite of a mastered KP.
        This ensures logical consistency in the student's knowledge state.
        
        Args:
            required_kps: Knowledge points required for the question
            missing_kps: Knowledge points the student is missing
            
        Returns:
            True if the combination is valid, False otherwise
        """
        mastered_kps = [kp for kp in required_kps if kp not in missing_kps]

        # Get all prerequisites of mastered KPs
        all_mastered_prereqs = self.get_all_prerequisites(mastered_kps)

        # Check if any missing KP is a prerequisite of mastered KPs
        for missing_kp in missing_kps:
            if missing_kp in all_mastered_prereqs:
                return False

        return True

    def compute_dense_state(self, missing_kps: List[str]) -> Tuple[List[str], List[str]]:
        """
        Compute dense knowledge state from missing KPs.
        
        Dense state includes all KPs that depend on the missing KPs
        (i.e., KPs that cannot be mastered without the missing prerequisites).
        
        Args:
            missing_kps: List of missing knowledge points
            
        Returns:
            Tuple of (mastered_kps, dense_missing_kps)
        """
        # Build reverse dependencies
        dependents: Dict[str, List[str]] = {kp: [] for kp in self.knowledge_points}
        for i, row in enumerate(self.adjacency_matrix):
            for j, dep in enumerate(row):
                if dep == 1:
                    dependents[self.knowledge_points[j]].append(self.knowledge_points[i])

        # Find all KPs that depend on missing KPs
        excluded: Set[str] = set()
        stack = list(missing_kps)
        while stack:
            cur = stack.pop()
            if cur in excluded:
                continue
            excluded.add(cur)
            for child in dependents.get(cur, []):
                if child not in excluded:
                    stack.append(child)

        dense_mastered = [kp for kp in self.knowledge_points if kp not in excluded]
        dense_missing = sorted(excluded)
        
        return dense_mastered, dense_missing

