import asyncio
import logging
import torch

from core.orchestrator import ProductionLMLMOrchestrator
from core.policies import PolicyEngine
from core.state import ToolDefinition
from memory import MemoryManager
from models.adapters import OpenAIAsyncAdapter, OllamaAsyncAdapter, MultiProviderRouter
from retrieval.rag import LMLMRAGPipeline, SimpleVectorStore
from tools.registry import ToolRegistry
from codex.sandbox import CODEXSandbox
from kernels.accelerator import FusedBiasGELU

logging.basicConfig(level=logging.INFO)


async def main():
    print("==================================================================")
    print("  LMLM (Large Multimodal Learning Model) Monolith Runtime v1.0    ")
    print("==================================================================")

    # 1. Test GPU CUDA Acceleration Layer
    print("\n[CUDA Layer Check]")
    hidden_dim = 256
    bias_gelu = FusedBiasGELU(hidden_dim=hidden_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bias_gelu.to(device)
    dummy_input = torch.randn(16, hidden_dim, device=device)
    cuda_out = bias_gelu(dummy_input)
    print(f"CUDA Accelerator Running on [{device}]. Tensor Shape: {cuda_out.shape}")

    # 2. Setup RAG & Memory
    print("\n[RAG Pipeline Ingestion]")
    vector_store = SimpleVectorStore()
    rag = LMLMRAGPipeline(vector_store)
    rag.ingest(
        source_id="manual",
        text="LMLM-CODEX isolates dynamic execution inside process sandboxes."
    )
    memory = MemoryManager()

    # 3. Setup Tools & LMLM-CODEX
    registry = ToolRegistry()
    sandbox = CODEXSandbox(timeout_seconds=5.0)

    async def run_codex(code: str):
        res = await sandbox.execute_python_script(code)
        return res.model_dump()

    registry.register(
        ToolDefinition(
            name="codex.execute",
            description="Executes code in CODEX sandbox.",
            input_schema={"code": "string"}
        ),
        run_codex
    )

    # 4. Multi-Provider Router with Fallback
    router = MultiProviderRouter(
        adapters=[
            OpenAIAsyncAdapter(model_name="gpt-4o"),
            OllamaAsyncAdapter(model_name="llama3")
        ]
    )

    # 5. Instantiate Orchestrator
    orchestrator = ProductionLMLMOrchestrator(
        model_adapter=router,
        tool_registry=registry,
        policy_engine=PolicyEngine(),
        memory_manager=memory
    )

    # 6. Execute End-to-End Orchestrator Loop
    print("\n[Executing 9-Phase LMLM Task]")
    task_input = "Execute array loop check in CODEX."
    trace = await orchestrator.execute_task(task_input)

    print("\n==================================================================")
    print("  LMLM Task Summary                                              ")
    print("==================================================================")
    print(f"Task ID: {trace.task_id}")
    print(f"Steps Completed: {len(trace.steps)}")
    print(f"Observation Count: {len(trace.observations)}")
    print(f"All Verifications Passed: {all(v.verified for v in trace.verifications)}")


if __name__ == "__main__":
    asyncio.run(main())
