"""
Manages running LangGraph streams per thread_id.
Enables interruption/cancellation of in-flight requests.
"""

import asyncio
from typing import Dict, Optional

class SessionManager:
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}

    def register_task(self, thread_id: str, task: asyncio.Task):
        """Register a new LangGraph stream task for a thread.
        Cancels any previous task for the same thread."""
        old_task = self.tasks.get(thread_id)
        if old_task and not old_task.done():
            old_task.cancel()
        self.tasks[thread_id] = task

    def get_task(self, thread_id: str) -> Optional[asyncio.Task]:
        """Get the current task for a thread."""
        return self.tasks.get(thread_id)

    def cancel(self, thread_id: str):
        """Cancel the running task for a thread."""
        task = self.tasks.get(thread_id)
        if task and not task.done():
            task.cancel()

    def cleanup(self, thread_id: str):
        """Remove a finished task."""
        self.tasks.pop(thread_id, None)


# Global singleton for the app
session_manager = SessionManager()
