import asyncio
import sys
import tempfile
from pathlib import Path
from pydantic import BaseModel


class ExecutionResult(BaseModel):
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time_ms: float


class CODEXSandbox:
    """Isolated process-level Python execution sandbox."""

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds

    async def execute_python_script(self, code_content: str) -> ExecutionResult:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "execution_payload.py"
            script_path.write_text(code_content, encoding="utf-8")
            start_time = asyncio.get_event_loop().time()

            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(script_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=temp_dir
                )

                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds
                )
                end_time = asyncio.get_event_loop().time()

                return ExecutionResult(
                    success=(proc.returncode == 0),
                    exit_code=proc.returncode or 0,
                    stdout=stdout.decode("utf-8"),
                    stderr=stderr.decode("utf-8"),
                    execution_time_ms=(end_time - start_time) * 1000
                )

            except asyncio.TimeoutError:
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Timed out after {self.timeout_seconds}s.",
                    execution_time_ms=self.timeout_seconds * 1000
                )
