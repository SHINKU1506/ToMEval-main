"""Run ToMEval chat prompts against local weights in-process with vLLM."""

from pathlib import Path
from typing import Any

from .client import LLMClient, LLMResponse


class VllmOfflineClient:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_path = Path(config["model_path"]).resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {self.model_path}")
        self.engine = None

    def batch_generate(self, prompts, desc="", system_prompts=None):
        if not prompts:
            return []

        from vllm import LLM, SamplingParams

        if self.engine is None:
            engine_options = {
                key: self.config[key] for key in (
                    "tensor_parallel_size", "max_model_len", "max_num_seqs",
                    "gpu_memory_utilization", "limit_mm_per_prompt", "seed",
                ) if key in self.config
            }
            self.engine = LLM(model=str(self.model_path), **engine_options)

        conversations = []
        for index, prompt in enumerate(prompts):
            system = (
                system_prompts[index] if system_prompts is not None
                else self.config.get("system_prompt")
            )
            messages = [{"role": "system", "content": system}] if system else []
            messages.append({"role": "user", "content": prompt})
            conversations.append(messages)

        thinking = self.config.get("enable_thinking", True)
        sampling = SamplingParams(
            temperature=self.config.get("temperature", 0.6),
            top_p=self.config.get("top_p", 0.95),
            top_k=self.config.get("top_k", 20),
            presence_penalty=self.config.get("presence_penalty", 2),
            max_tokens=self.config.get("max_tokens", 32768),
        )
        outputs = self.engine.chat(
            conversations, sampling, chat_template_kwargs={"enable_thinking": thinking}
        )
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned a different number of outputs than input prompts.")

        results = []
        for output in outputs:
            text = output.outputs[0].text
            content, reasoning = LLMClient._split_think_content(text) if thinking else (text, "")
            results.append(LLMResponse(content=content or None, reasoning=reasoning))
        return results
