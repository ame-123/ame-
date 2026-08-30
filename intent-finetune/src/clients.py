"""测评后端：env 里现有路由模型，以及微调后的 OpenAI 兼容服务。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.labels import load_sft_instruction, parse_model_output
from src.paths import SFT_PROMPT_PATH, TRAVEL_ENV, ensure_travel_backend_on_path


@dataclass
class Prediction:
    backend: str
    model_name: str | None
    intent: str | None
    order_id: str | None
    slots: dict[str, Any] | None
    raw_text: str
    used_model: bool
    parse_ok: bool
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _load_env() -> None:
    ensure_travel_backend_on_path()
    from config.settings import load_course_env

    os.environ.setdefault("AGENT_COURSE_ENV", str(TRAVEL_ENV))
    load_course_env()


def _chat_openai(*, model: str, base_url: str, api_key: str, system_prompt: str, user_message: str) -> str:
    from langchain_core.messages import SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        temperature=0,
            timeout=20,
        max_retries=2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            ("human", "{user_message}"),
        ]
    )
    chain = prompt | chat | StrOutputParser()
    return str(chain.invoke({"user_message": user_message}))


def _production_route_prompt() -> str:
    from prompts.loader import prompt_manager

    fragments = prompt_manager.select_fragments({"phase": "route"})
    return prompt_manager.render_system_prompt(fragments)


def _from_text(backend: str, model_name: str | None, text: str, *, used_model: bool, error: str | None = None) -> Prediction:
    parsed = parse_model_output(text) if text else None
    return Prediction(
        backend=backend,
        model_name=model_name,
        intent=None if parsed is None else parsed["intent"],
        order_id=None if parsed is None else parsed.get("order_id"),
        slots=None if parsed is None else parsed.get("slots"),
        raw_text=text or "",
        used_model=used_model,
        parse_ok=parsed is not None,
        error=error,
    )


class EnvRouteClient:
    """和线上 RouteModelClient 同一套 env：AGENT_OPENAI_MODEL + 生产 route prompt。"""

    name = "env_router"

    def __init__(self) -> None:
        _load_env()
        from config.settings import openai_base_url, openai_model_name

        self.model_name = openai_model_name()
        self.base_url = openai_base_url()
        self.api_key = os.getenv("AGENT_OPENAI_API_KEY") or ""
        self.system_prompt = _production_route_prompt()

    def predict(self, user_message: str) -> Prediction:
        from config.settings import api_key_is_missing

        if api_key_is_missing(self.api_key):
            return _from_text(self.name, self.model_name, "", used_model=False, error="api_key_missing")
        try:
            text = _chat_openai(
                model=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                system_prompt=self.system_prompt,
                user_message=user_message,
            )
            return _from_text(self.name, self.model_name, text, used_model=True)
        except Exception as exc:
            return _from_text(self.name, self.model_name, "", used_model=False, error=exc.__class__.__name__)


class EnvClassifierClient:
    """course.env 里的 AGENT_CLASSIFIER_MODEL（当前线上路由未接它，单独作为对照）。"""

    name = "env_classifier"

    def __init__(self) -> None:
        _load_env()
        from config.settings import openai_base_url

        self.model_name = os.getenv("AGENT_CLASSIFIER_MODEL") or ""
        self.base_url = openai_base_url()
        self.api_key = os.getenv("AGENT_OPENAI_API_KEY") or ""
        self.system_prompt = _production_route_prompt()

    def predict(self, user_message: str) -> Prediction:
        from config.settings import api_key_is_missing

        if not self.model_name:
            return _from_text(self.name, None, "", used_model=False, error="classifier_model_missing")
        if api_key_is_missing(self.api_key):
            return _from_text(self.name, self.model_name, "", used_model=False, error="api_key_missing")
        try:
            text = _chat_openai(
                model=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                system_prompt=self.system_prompt,
                user_message=user_message,
            )
            return _from_text(self.name, self.model_name, text, used_model=True)
        except Exception as exc:
            return _from_text(self.name, self.model_name, "", used_model=False, error=exc.__class__.__name__)


class FinetunedClient:
    """微调后的 OpenAI 兼容接口。未启动服务时预测会标 skipped。"""

    name = "finetuned"

    def __init__(self) -> None:
        _load_env()
        self.model_name = os.getenv("FINETUNED_MODEL") or os.getenv("ROUTER_FINETUNED_MODEL") or "router-lora"
        self.base_url = (os.getenv("FINETUNED_BASE_URL") or os.getenv("ROUTER_FINETUNED_BASE_URL") or "").rstrip("/")
        self.api_key = os.getenv("FINETUNED_API_KEY") or os.getenv("AGENT_OPENAI_API_KEY") or "sk-local"
        self.system_prompt = load_sft_instruction(SFT_PROMPT_PATH)

    def available(self) -> bool:
        return bool(self.base_url)

    def predict(self, user_message: str) -> Prediction:
        if not self.available():
            return _from_text(self.name, self.model_name, "", used_model=False, error="finetuned_endpoint_missing")
        try:
            text = _chat_openai(
                model=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                system_prompt=self.system_prompt,
                user_message=user_message,
            )
            return _from_text(self.name, self.model_name, text, used_model=True)
        except Exception as exc:
            return _from_text(self.name, self.model_name, "", used_model=False, error=exc.__class__.__name__)


class LocalHFClient:
    """本机 HuggingFace 推理，不走硅基流动。默认 Qwen2.5-3B-Instruct。"""

    name = "local"
    _tokenizer = None
    _model = None
    _loaded_name: str | None = None

    def __init__(self) -> None:
        self.model_name = os.getenv("LOCAL_ROUTER_MODEL") or "Qwen/Qwen2.5-3B-Instruct"
        self.adapter = (os.getenv("LOCAL_ROUTER_ADAPTER") or "").strip()
        self.system_prompt = load_sft_instruction(SFT_PROMPT_PATH)
        self._cache_key = f"{self.model_name}::{self.adapter}"
        self._load()

    def _local_hf_snapshot(self, repo_id: str) -> str | None:
        cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
        snaps = cache / "snapshots"
        if not snaps.is_dir():
            return None
        for snap in snaps.iterdir():
            if not (snap / "config.json").is_file():
                continue
            shards = list(snap.glob("model*.safetensors"))
            if shards and all(p.stat().st_size > 1_000_000 for p in shards):
                return str(snap)
        return None

    def _resolve_source(self) -> str:
        name = self.model_name
        if os.path.isdir(name):
            return name
        local = self._local_hf_snapshot(name)
        if local:
            print(f"[local] using cached snapshot {local}", flush=True)
            return local
        try:
            from modelscope import snapshot_download

            print(f"[local] resolving {name} via ModelScope...", flush=True)
            path = snapshot_download(model_id=name)
            print(f"[local] ModelScope path: {path}", flush=True)
            return path
        except Exception as exc:
            print(f"[local] ModelScope unavailable ({exc}); using HuggingFace id", flush=True)
            return name

    def _load(self) -> None:
        if LocalHFClient._model is not None and LocalHFClient._loaded_name == self._cache_key:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # hf-mirror + xet 会 401；国内优先 ModelScope。
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        source = self._resolve_source()
        local_only = os.path.isdir(source)
        print(f"[local] loading {source} on cuda...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            source, trust_remote_code=True, local_files_only=local_only
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        load_kw = {
            "trust_remote_code": True,
            "device_map": "cuda" if torch.cuda.is_available() else "cpu",
            "local_files_only": local_only,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype, **load_kw)
        except TypeError:
            load_kw.pop("dtype", None)
            model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, **load_kw)
        if self.adapter:
            from peft import PeftModel

            print(f"[local] loading adapter {self.adapter}", flush=True)
            model = PeftModel.from_pretrained(model, self.adapter)
        model.eval()
        LocalHFClient._tokenizer = tokenizer
        LocalHFClient._model = model
        LocalHFClient._loaded_name = self._cache_key
        print("[local] ready", flush=True)

    def predict(self, user_message: str) -> Prediction:
        import torch

        tokenizer = LocalHFClient._tokenizer
        model = LocalHFClient._model
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return _from_text(self.name, self.model_name, text, used_model=True)
        except Exception as exc:
            return _from_text(self.name, self.model_name, "", used_model=False, error=exc.__class__.__name__)


def build_client(name: str):
    mapping = {
        "env": EnvRouteClient,
        "env_router": EnvRouteClient,
        "env_classifier": EnvClassifierClient,
        "finetuned": FinetunedClient,
        "local": LocalHFClient,
    }
    if name not in mapping:
        raise ValueError(f"unknown backend {name}, expected {sorted(mapping)}")
    return mapping[name]()
