"""LLM adapter — turn a free Chinese intent into a slot table.

The model only *decides*; it never writes the final prompt string.  Everything
here speaks the OpenAI ``/chat/completions`` wire format on purpose: the same
client serves a hosted endpoint (LM Studio, vLLM, any aggregator) and a local
``llama-server`` launched on the GGUF produced by M4.
"""

from .client import ChatClient, ChatConfig, resolve_config
from .pipeline import PipelineResult, run_intent
from .slotfill import FillResult, fill_slots

__all__ = [
    "ChatClient",
    "ChatConfig",
    "resolve_config",
    "FillResult",
    "fill_slots",
    "PipelineResult",
    "run_intent",
]
