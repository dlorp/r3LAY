"""Diagnostic flowchart engine for systematic troubleshooting.

Flowcharts guide users through diagnostic procedures with:
- Yes/No questions
- Test procedures with expected results
- Branching logic based on outcomes
- Terminal nodes (diagnosis complete)
- Tool/equipment requirements per step

Flowcharts are defined in YAML for easy editing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml


class NodeType(str, Enum):
    """Flowchart node types."""

    QUESTION = "question"  # Yes/No decision
    TEST = "test"  # Perform test, compare result
    DIAGNOSIS = "diagnosis"  # Terminal node (conclusion)
    REFERENCE = "reference"  # External link/document


@dataclass
class FlowchartNode:
    """Single node in a diagnostic flowchart.

    Attributes:
        id: Unique node identifier (e.g., "start", "cranks_check_fuel")
        type: Node type (question, test, diagnosis, reference)
        text: Question/instruction text
        yes_next: Next node if yes/pass (None for terminal nodes)
        no_next: Next node if no/fail (None for terminal nodes)
        tools_required: Tools needed for this step
        notes: Additional context/tips
        obd_codes: Related DTC codes
    """

    id: str
    type: NodeType
    text: str
    yes_next: str | None = None
    no_next: str | None = None
    tools_required: list[str] = field(default_factory=list)
    notes: str | None = None
    obd_codes: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Convert type string to enum if needed."""
        if isinstance(self.type, str):
            self.type = NodeType(self.type)

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (diagnosis/reference)."""
        return self.type in {NodeType.DIAGNOSIS, NodeType.REFERENCE}


class DiagnosticFlowchart:
    """Represents an interactive diagnostic procedure.

    Loads flowchart from YAML file and provides navigation/query methods.

    Attributes:
        name: Flowchart name
        description: Brief description
        symptoms: List of applicable symptoms
        difficulty: Difficulty level (beginner/intermediate/advanced)
        time_estimate: Estimated completion time
        start_node: ID of starting node
        nodes: Dictionary of nodes (id -> FlowchartNode)

    Example:
        >>> flowchart = DiagnosticFlowchart(Path("no-start.yaml"))
        >>> print(flowchart.name)
        Engine Won't Start - Basic Diagnosis
        >>> start = flowchart.get_node(flowchart.start_node)
        >>> print(start.text)
        Does the engine crank when you turn the key to START?
    """

    def __init__(self, flowchart_file: Path):
        self.file = flowchart_file

        with open(flowchart_file) as f:
            data = yaml.safe_load(f)

        self.name: str = data["name"]
        self.description: str = data["description"]
        self.symptoms: list[str] = data.get("symptoms", [])
        self.difficulty: str = data.get("difficulty", "intermediate")
        self.time_estimate: str = data.get("time_estimate", "unknown")
        self.tools_required: list[str] = data.get("tools_required", [])

        # Parse nodes
        self.nodes: dict[str, FlowchartNode] = {}
        for node_data in data.get("nodes", []):
            node = FlowchartNode(**node_data)
            self.nodes[node.id] = node

        self.start_node: str = data.get("start_node", "start")

        # Validate
        if self.start_node not in self.nodes:
            raise ValueError(f"Start node '{self.start_node}' not found in flowchart")

    def get_node(self, node_id: str) -> FlowchartNode | None:
        """Get node by ID.

        Args:
            node_id: Node identifier

        Returns:
            FlowchartNode if found, None otherwise
        """
        return self.nodes.get(node_id)

    def is_terminal(self, node_id: str) -> bool:
        """Check if node is a terminal diagnosis.

        Args:
            node_id: Node identifier

        Returns:
            True if node is terminal (diagnosis/reference), False otherwise
        """
        node = self.get_node(node_id)
        return node.is_terminal() if node else False

    def get_path(self, start: str, end: str) -> list[str]:
        """Get diagnostic path from start to end node (BFS).

        Args:
            start: Starting node ID
            end: Target node ID

        Returns:
            List of node IDs from start to end (inclusive)
            Empty list if no path exists

        Example:
            >>> path = flowchart.get_path("start", "battery_low")
            >>> for node_id in path:
            ...     node = flowchart.get_node(node_id)
            ...     print(f"{node_id}: {node.text}")
        """
        if start not in self.nodes or end not in self.nodes:
            return []

        # BFS to find path
        queue = [(start, [start])]
        visited = {start}

        while queue:
            current, path = queue.pop(0)

            if current == end:
                return path

            node = self.nodes[current]

            # Add yes branch
            if node.yes_next and node.yes_next not in visited:
                visited.add(node.yes_next)
                queue.append((node.yes_next, path + [node.yes_next]))

            # Add no branch
            if node.no_next and node.no_next not in visited:
                visited.add(node.no_next)
                queue.append((node.no_next, path + [node.no_next]))

        return []  # No path found

    def get_all_tools(self) -> set[str]:
        """Get set of all tools mentioned in flowchart.

        Returns:
            Set of tool names required by any node
        """
        tools = set(self.tools_required)
        for node in self.nodes.values():
            tools.update(node.tools_required)
        return tools

    def __len__(self) -> int:
        """Return number of nodes in flowchart."""
        return len(self.nodes)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DiagnosticFlowchart(name='{self.name}', nodes={len(self.nodes)})"


class FlowchartSession:
    """Interactive flowchart navigation session.

    Tracks current position and history for yes/no/back navigation.

    Attributes:
        flowchart: The diagnostic flowchart being navigated
        current_node: Current node ID
        history: List of visited node IDs

    Example:
        >>> flowchart = DiagnosticFlowchart(Path("no-start.yaml"))
        >>> session = FlowchartSession(flowchart)
        >>> print(session.get_current_node().text)
        Does the engine crank when you turn the key to START?
        >>> session.answer_yes()
        >>> print(session.current_node)
        cranks_check_fuel
    """

    def __init__(self, flowchart: DiagnosticFlowchart):
        self.flowchart = flowchart
        self.current_node = flowchart.start_node
        self.history: list[str] = [flowchart.start_node]

    def get_current_node(self) -> FlowchartNode:
        """Get current node object.

        Returns:
            Current FlowchartNode

        Raises:
            KeyError: If current node ID is invalid
        """
        node = self.flowchart.get_node(self.current_node)
        if not node:
            raise KeyError(f"Invalid node ID: {self.current_node}")
        return node

    def answer_yes(self) -> None:
        """Navigate to yes branch.

        Raises:
            ValueError: If current node has no yes branch
        """
        node = self.get_current_node()
        if not node.yes_next:
            raise ValueError(f"Node {self.current_node} has no yes branch")

        self.current_node = node.yes_next
        self.history.append(self.current_node)

    def answer_no(self) -> None:
        """Navigate to no branch.

        Raises:
            ValueError: If current node has no no branch
        """
        node = self.get_current_node()
        if not node.no_next:
            raise ValueError(f"Node {self.current_node} has no no branch")

        self.current_node = node.no_next
        self.history.append(self.current_node)

    def go_back(self) -> None:
        """Go back to previous node.

        Raises:
            ValueError: If already at start node
        """
        if len(self.history) <= 1:
            raise ValueError("Already at start node, cannot go back")

        self.history.pop()
        self.current_node = self.history[-1]

    def restart(self) -> None:
        """Restart from beginning."""
        self.current_node = self.flowchart.start_node
        self.history = [self.flowchart.start_node]

    def is_complete(self) -> bool:
        """Check if we've reached a terminal node.

        Returns:
            True if current node is terminal (diagnosis/reference)
        """
        return self.get_current_node().is_terminal()

    def get_breadcrumbs(self) -> list[str]:
        """Get breadcrumb trail of visited nodes.

        Returns:
            List of node IDs in order visited
        """
        return self.history.copy()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"FlowchartSession(current={self.current_node}, steps={len(self.history)})"
