from typing import Any, Dict, List


class MemoryManager:
    """Segregated memory subsystem separating working context from episodic recall."""

    def __init__(self):
        self.working_memory: Dict[str, List[Dict[str, Any]]] = {}
        self.persistent_episodic: List[Dict[str, Any]] = []

    def update_working(self, task_id: str, entry: Dict[str, Any]):
        if task_id not in self.working_memory:
            self.working_memory[task_id] = []
        self.working_memory[task_id].append(entry)

    def commit_to_persistent(self, task_id: str, trace_summary: Dict[str, Any]):
        self.persistent_episodic.append({
            "task_id": task_id,
            "summary": trace_summary
        })
        self.working_memory.pop(task_id, None)
