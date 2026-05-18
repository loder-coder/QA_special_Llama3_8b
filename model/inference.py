from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model.prompt import (
    GenerationConfig,
    SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
    build_prompt,
    build_user_prompt,
    escape_prompt_section,
)


class LlamaAnswerGenerator:
    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3-8b",
        adapter_path: str | None = "artifacts/llama_lora",
        load_in_4bit: bool = True,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    @staticmethod
    def _escape_prompt_section(value: str) -> str:
        return escape_prompt_section(value)

    @staticmethod
    def build_user_prompt(question: str, context: str = "", category: str = "", language: str = "ko") -> str:
        return build_user_prompt(question=question, context=context, category=category, language=language)

    @staticmethod
    def build_prompt(question: str, context: str = "", category: str = "", language: str = "ko") -> str:
        return build_prompt(question=question, context=context, category=category, language=language)

    def generate(
        self,
        question: str,
        context: str = "",
        category: str = "",
        language: str = "ko",
        config: GenerationConfig | None = None,
    ) -> str:
        generation_config = config or GenerationConfig()
        prompt = self.build_prompt(question=question, context=context, category=category, language=language)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                do_sample=generation_config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return generated.strip()
