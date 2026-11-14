"""Minimal shared helpers for the standardized QA evaluation flow."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlsplit

import yaml

from src.dataloader import load_dataset

logger = logging.getLogger(__name__)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _load_api_profiles(
    path: Path, stage: str, datasets: List[str], use_target_profile: bool
) -> Dict[str, Any]:
    import tomllib

    with path.open("rb") as file:
        profiles = tomllib.load(file)
    issues = []

    def profile(name: str, values: Dict[str, Any], needs_connection: bool) -> Dict[str, Any]:
        required = ("model_name", "api_url", "api_key") if needs_connection else ("model_name",)
        for key in required:
            value = values.get(key)
            if not isinstance(value, str) or not value.strip():
                issues.append(f"{name}.{key}")
            elif key == "api_url":
                url = urlsplit(value)
                if url.scheme not in {"http", "https"} or not url.hostname:
                    issues.append(f"{name}.{key} (expected an HTTP(S) base URL)")
        return dict(values)

    options: Dict[str, Any] = {}
    if use_target_profile:
        options["target"] = profile(
            "target", profiles.get("target", {}), stage in {"predict", "all"}
        )
    if stage in {"metric", "all"} and "SoMBench" in datasets:
        sombench = profiles.get("sombench", {})
        options["sombench_judges"] = {
            name: profile(f"sombench.{name}", sombench.get(name, {}), True)
            for name in ("judge1", "judge2")
        }
    if issues:
        raise ValueError(f"Fill in {path}: {', '.join(issues)}")
    return options


def load_experiment_config(config_path: str, dataset: str | None = None) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as file:
        config = _expand_env_vars(yaml.safe_load(file) or {})

    llm_config = dict(config.get("llm", {}))
    datasets = list(config.get("datasets") or [])
    stage = config.get("stage", "all")
    api_options = {}
    if config.get("api_config"):
        api_path = Path(config_path).resolve().parent / config["api_config"]
        use_target_profile = llm_config.get("backend") != "vllm" and (
            stage in {"predict", "all"} or not llm_config.get("model_name")
        )
        api_options = _load_api_profiles(
            api_path, stage, [dataset] if dataset else datasets, use_target_profile
        )
        llm_config.update(api_options.pop("target", {}))
        for name, judge in api_options.get("sombench_judges", {}).items():
            if config.get("judge_cache_path"):
                judge["cache_dir"] = str(Path(config["judge_cache_path"]) / name)
            if config.get("judge_request_timeout"):
                judge["request_timeout"] = config["judge_request_timeout"]
            if config.get("judge_abort_on_rate_limit"):
                judge["abort_on_rate_limit"] = True
            if config.get("judge_max_workers"):
                judge["max_workers"] = int(config["judge_max_workers"])
    repeats = config.get("repeats", 1)
    protocol = config.get("protocol")

    # agent(智能体评测)模式:predictor: agent 时走远程黑盒 HTTP 端点,不使用采样协议 protocol。
    # agent 内部代码自写、协议不限、可多轮/多次调 model;调 model 用的凭证(我们部署的统一后端)
    # 随每条请求体的 model 字段下发给 agent。
    predictor = config.get("predictor")
    agent_mode = predictor == "agent"
    agent_config = dict(config.get("agent", {}))
    if agent_mode:
        # agent 模式与采样协议互斥:agent 内部自定采样,protocol 权威表不再适用。
        if protocol is not None:
            logger.info("predictor=agent:忽略 protocol,采样由 agent 内部决定。")
            protocol = None
        # repeats 固定为 1(agent 若要多次采样/投票由其内部实现)。
        repeats = 1

    # 设了协议:采样参数(temperature/top_p/max_tokens/enable_thinking)由协议权威表覆盖，
    # repeats 取协议的 n_samples(direct/direct_think/cot→1, del_tom→8)，忽略 config 的 repeats。
    if protocol is not None:
        from src.evaluation.protocols import llm_overrides_for, repeats_for

        llm_config.update(llm_overrides_for(protocol))
        derived_repeats = repeats_for(protocol, repeats)
        if derived_repeats != repeats:
            logger.info(
                f"Protocol '{protocol}': repeats overridden {repeats} -> {derived_repeats} (= n_samples)."
            )
        repeats = derived_repeats
        logger.info(f"Protocol '{protocol}': llm sampling overridden to {llm_overrides_for(protocol)}.")

    return {
        "llm_config": llm_config,
        "repeats": repeats,
        "max_samples": config.get("max_samples", 0),
        "normalized_datasets_path": config.get("normalized_datasets_path", "datasets_normalized"),
        "results_path": config.get("results_path", "results"),
        "figures_path": config.get("figures_path", "figures"),
        "logs_path": config.get("logs_path", "logs"),
        # judge 参数不再由 experiment_config 管理，移到各数据集 tasks/<DS>/config.yaml 的 judge1/judge2 段。
        "protocol": protocol,
        # stage 与 datasets 由配置驱动:stage 控制 predict/metric/all,datasets 供 run_eval 批量评测。
        "stage": stage,
        "datasets": datasets,
        # agent(智能体)评测:agent_mode 标识走远程黑盒 HTTP 端点;agent_config 为 agent 段(api_url/api_key/超时/并发)。
        "agent_mode": agent_mode,
        "agent_config": agent_config,
        **api_options,
    }


def create_llm_client(llm_config: Dict[str, Any]) -> Any:
    config = _expand_env_vars(llm_config.copy())
    from src.llm import StructureClient

    client = StructureClient.from_config(config)
    if config.get("cache_dir"):
        client.cache_dir = Path(config["cache_dir"]).resolve()
    if config.get("request_timeout"):
        client.request_timeout = float(config["request_timeout"])
    client.abort_on_rate_limit = bool(config.get("abort_on_rate_limit", False))
    return client


def create_content_client(llm_config: Dict[str, Any]) -> Any:
    config = _expand_env_vars(llm_config.copy())
    if config.get("backend") == "vllm":
        from src.llm.vllm_offline_client import VllmOfflineClient

        return VllmOfflineClient(config)
    from src.llm import ContentClient

    return ContentClient.from_config(config)


def load_and_limit_data(
    subset: str,
    datasets_root: str = "datasets_normalized",
    max_samples: int = 0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    import random

    data = load_dataset(subset, datasets_root=datasets_root)
    if max_samples > 0:
        random.seed(seed)
        data = random.sample(data, min(max_samples, len(data)))
    return data
