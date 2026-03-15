#!/usr/bin/env python3
from __future__ import annotations

"""
Interactive Ollama stress test harness.

Run:
  python ollama_stress_test_interactive.py

What it does:
- Uses embedded defaults for base URL and output directory
- Prompts you to choose a primary model from `ollama list`
- Prompts you to choose an optional worker model from `ollama list`
- Prompts for keep-alive with a default of 30m
- Runs cold/warm, concurrency, and mixed-residency tests
- Prints colored human-readable logs
- Writes a final report JSON
"""

import concurrent.futures
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_URL = "http://192.168.0.213:11434"
OUTPUT_DIR = Path(__file__).resolve().parent / "stress-results"
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_NUM_PREDICT = 700
HTTP_TIMEOUT_S = 1200

print_lock = threading.Lock()


def enable_ansi() -> None:
    if os.name == "nt":
        try:
            os.system("")
        except Exception:
            pass


enable_ansi()


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_CYAN = "\033[96m"


def color(text: str, *codes: str) -> str:
    return "".join(codes) + text + C.RESET


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def ns_to_s(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / 1_000_000_000, 4)


def fmt_s(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}s"


def fmt_num(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den in (None, 0):
        return None
    return num / den


def emit(msg: str = "") -> None:
    with print_lock:
        print(msg, flush=True)


def rule(title: str) -> None:
    line = "=" * 22
    emit(color(f"{line} {title} {line}", C.CYAN, C.BOLD))


def panel(title: str, body_lines: List[str]) -> None:
    width = max([len(title)] + [len(x) for x in body_lines] + [56])
    top = f"╔{'═' * (width + 2)}╗"
    bot = f"╚{'═' * (width + 2)}╝"
    emit(color(top, C.CYAN))
    emit(color(f"║ {title.ljust(width)} ║", C.CYAN, C.BOLD))
    for line in body_lines:
        emit(color(f"║ {line.ljust(width)} ║", C.CYAN))
    emit(color(bot, C.CYAN))


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = HTTP_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post_json(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            url=f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} calling {endpoint}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error calling {endpoint}: {e}") from e

    def show_model(self, model: str) -> Dict[str, Any]:
        return self._post_json("/api/show", {"model": model})

    def generate(
        self,
        model: str,
        prompt: str = "",
        *,
        keep_alive: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if options:
            payload["options"] = options
        return self._post_json("/api/generate", payload)

    def preload(self, model: str, keep_alive: Any) -> Dict[str, Any]:
        return self.generate(model=model, prompt="", keep_alive=keep_alive)

    def unload(self, model: str) -> Dict[str, Any]:
        return self.generate(model=model, prompt="", keep_alive=0)


BASE_CONTEXT = (
    "Project notes:\n"
    "- The system uses a primary orchestrator model for planning and synthesis.\n"
    "- Worker models may handle coding, extraction, summarization, and review.\n"
    "- The goal is to assess latency, concurrency, and long-context performance.\n"
    "- The environment includes a control host, shared storage, and a GPU processing node.\n"
    "- The test should reveal whether a heavy orchestrator model remains responsive under realistic load.\n"
    "- Final responses should be structured and concise."
)

LONG_PARAGRAPH = (
    "A reliable orchestrator should read messy notes, identify priorities, choose what must be done serially versus in parallel, "
    "and explain why. It should preserve constraints, detect hidden dependencies, and avoid pretending that every task deserves "
    "the same expensive level of reasoning. A good benchmark for orchestration is not raw speed alone, but whether the model stays "
    "usable when juggling synthesis, routing, and long retrieval context at the same time. "
)


def build_context(repetitions: int) -> str:
    return (BASE_CONTEXT + "\n\n" + (LONG_PARAGRAPH * repetitions)).strip()


def prompt_short() -> str:
    return (
        "You are benchmarking orchestration quality. Read the short project notes and return a JSON object with keys: "
        "goal, risks, parallelizable_tasks, serial_tasks, and recommendation. Keep the values concise.\n\n"
        f"{build_context(2)}"
    )


def prompt_medium() -> str:
    return (
        "Analyze the following project notes and produce a compact plan with three sections: summary, bottlenecks, and next steps. "
        "Each section must use short bullet points.\n\n"
        f"{build_context(25)}"
    )


def prompt_long() -> str:
    return (
        "You are the main orchestrator. Read the notes below and produce a structured response with: executive summary, system risks, "
        "recommended routing policy, and a short action plan. Use precise language and do not repeat points.\n\n"
        f"{build_context(120)}"
    )


def prompt_xlong() -> str:
    return (
        "Perform long-context synthesis. The text below simulates retrieved notes, benchmark fragments, environment facts, and prior reasoning. "
        "Return: 1) major observations, 2) concurrency risks, 3) suggested operating profile, 4) whether this model should be used as the "
        "default sub-agent model. Be direct.\n\n"
        f"{build_context(260)}"
    )


def prompt_worker() -> str:
    return (
        "Review this tiny task and answer with exactly three bullet points: likely task type, best worker role, and one risk.\n\n"
        f"{build_context(6)}"
    )


PROMPTS = {
    "short": prompt_short,
    "medium": prompt_medium,
    "long": prompt_long,
    "xlong": prompt_xlong,
    "worker": prompt_worker,
}


def shape_metrics(response: Dict[str, Any], started_at: float, finished_at: float) -> Dict[str, Any]:
    total_s = ns_to_s(response.get("total_duration"))
    load_s = ns_to_s(response.get("load_duration"))
    prompt_eval_s = ns_to_s(response.get("prompt_eval_duration"))
    eval_s = ns_to_s(response.get("eval_duration"))
    wall_s = round(finished_at - started_at, 4)

    prompt_eval_count = response.get("prompt_eval_count")
    eval_count = response.get("eval_count")
    prompt_tps = safe_div(float(prompt_eval_count), prompt_eval_s) if prompt_eval_count is not None else None
    gen_tps = safe_div(float(eval_count), eval_s) if eval_count is not None else None
    ttft_proxy_s = None
    if load_s is not None or prompt_eval_s is not None:
        ttft_proxy_s = round((load_s or 0) + (prompt_eval_s or 0), 4)

    return {
        "wall_time_s": wall_s,
        "total_duration_s": total_s,
        "load_duration_s": load_s,
        "prompt_eval_duration_s": prompt_eval_s,
        "eval_duration_s": eval_s,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "prompt_tokens_per_s": round(prompt_tps, 2) if prompt_tps is not None else None,
        "gen_tokens_per_s": round(gen_tps, 2) if gen_tps is not None else None,
        "ttft_proxy_s": ttft_proxy_s,
        "done": response.get("done"),
        "done_reason": response.get("done_reason"),
    }


def summarize_metrics(m: Dict[str, Any]) -> str:
    return (
        f"wall={fmt_s(m['wall_time_s'])} | total={fmt_s(m['total_duration_s'])} | "
        f"load={fmt_s(m['load_duration_s'])} | prompt={fmt_s(m['prompt_eval_duration_s'])} | "
        f"gen={fmt_s(m['eval_duration_s'])} | in_tps={fmt_num(m['prompt_tokens_per_s'])} | "
        f"out_tps={fmt_num(m['gen_tokens_per_s'])} | ttft≈{fmt_s(m['ttft_proxy_s'])}"
    )


def list_models() -> List[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as e:
        raise RuntimeError("`ollama` command was not found in PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"`ollama list` failed: {e.stderr or e.stdout}") from e

    lines = [line.rstrip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError("No models returned by `ollama list`.")

    models: List[str] = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    if not models:
        raise RuntimeError("Could not parse model names from `ollama list`.")
    return models


def choose_model(models: List[str], prompt_text: str, allow_skip: bool = False) -> Optional[str]:
    emit(color(prompt_text, C.BOLD, C.CYAN))
    for idx, model in enumerate(models, start=1):
        emit(f"  {color(str(idx).rjust(2), C.BOLD, C.YELLOW)}. {color(model, C.WHITE)}")
    if allow_skip:
        emit(f"  {color(' 0', C.BOLD, C.YELLOW)}. {color('None / skip mixed residency test', C.BRIGHT_BLACK)}")

    while True:
        raw = input(color("Select number: ", C.BOLD, C.GREEN)).strip()
        if allow_skip and raw == "0":
            return None
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(models):
                return models[n - 1]
        emit(color("Invalid selection. Enter a number from the list.", C.RED))


def prompt_keep_alive(default_value: str) -> str:
    emit(color(f"*Keep Alive* set to {default_value}. Press Enter to begin stress test or provide updated value:", C.BOLD, C.MAGENTA))
    raw = input(color("Keep-alive: ", C.BOLD, C.GREEN)).strip()
    return raw or default_value


class StressTester:
    def __init__(self, client: OllamaClient, primary_model: str, worker_model: Optional[str], keep_alive: str):
        self.client = client
        self.primary_model = primary_model
        self.worker_model = worker_model
        self.keep_alive = keep_alive
        self.report: Dict[str, Any] = {
            "started_at": now_iso(),
            "base_url": client.base_url,
            "model": primary_model,
            "worker_model": worker_model,
            "keep_alive": keep_alive,
            "tests": [],
            "summary": {},
            "environment": {},
        }

    def _options(self) -> Dict[str, Any]:
        return {"temperature": DEFAULT_TEMPERATURE, "num_predict": DEFAULT_NUM_PREDICT}

    def inspect_model(self, model: str) -> Dict[str, Any]:
        try:
            details = self.client.show_model(model)
            return {
                "model": model,
                "capabilities": details.get("capabilities"),
                "details": details.get("details"),
                "parameters": details.get("parameters"),
                "modified_at": details.get("modified_at"),
            }
        except Exception as e:
            return {"model": model, "error": str(e)}

    def run_single(self, name: str, model: str, prompt_name: str, *, preload: bool, force_cold: bool, keep_alive: Any) -> None:
        prompt = PROMPTS[prompt_name]()
        emit(color(f"→ {name}", C.BOLD, C.CYAN) + f"  model={color(model, C.WHITE)}  prompt={color(prompt_name, C.WHITE)}")
        if force_cold:
            emit(color("   unloading model first (keep_alive=0) to force a cold path...", C.BRIGHT_BLACK))
            try:
                self.client.unload(model)
            except Exception as e:
                emit(color(f"   unload note: {e}", C.YELLOW))
            time.sleep(0.75)
        if preload:
            emit(color(f"   preloading model (keep_alive={keep_alive})...", C.BRIGHT_BLACK))
            self.client.preload(model, keep_alive=keep_alive)
            time.sleep(0.5)

        started = time.perf_counter()
        response = self.client.generate(model=model, prompt=prompt, keep_alive=keep_alive, options=self._options())
        finished = time.perf_counter()
        metrics = shape_metrics(response, started, finished)
        emit("   " + color(summarize_metrics(metrics), C.WHITE))
        self.report["tests"].append({
            "name": name,
            "kind": "single",
            "model": model,
            "prompt_name": prompt_name,
            "prompt_chars": len(prompt),
            "preload": preload,
            "force_cold": force_cold,
            "keep_alive": keep_alive,
            "started_at": now_iso(),
            "metrics": metrics,
            "response_preview": (response.get("response") or "")[:1200],
            "raw_response": response,
        })

    def run_concurrent(self, name: str, model: str, prompt_name: str, concurrency: int, *, keep_alive: Any) -> None:
        prompt = PROMPTS[prompt_name]()
        emit(color(f"→ {name}", C.BOLD, C.MAGENTA) + f"  model={color(model, C.WHITE)}  prompt={color(prompt_name, C.WHITE)}  concurrency={color(str(concurrency), C.WHITE)}")
        self.client.preload(model, keep_alive=keep_alive)
        time.sleep(0.4)

        def worker_call(index: int) -> Dict[str, Any]:
            started = time.perf_counter()
            response = self.client.generate(model=model, prompt=prompt, keep_alive=keep_alive, options=self._options())
            finished = time.perf_counter()
            return {
                "index": index,
                "metrics": shape_metrics(response, started, finished),
                "response_preview": (response.get("response") or "")[:500],
                "raw_response": response,
            }

        batch_started = time.perf_counter()
        runs: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(worker_call, i + 1) for i in range(concurrency)]
            for fut in concurrent.futures.as_completed(futures):
                runs.append(fut.result())
        batch_finished = time.perf_counter()

        wall_times = [r["metrics"]["wall_time_s"] for r in runs]
        gen_tps = [r["metrics"]["gen_tokens_per_s"] for r in runs if r["metrics"]["gen_tokens_per_s"] is not None]
        ttfts = [r["metrics"]["ttft_proxy_s"] for r in runs if r["metrics"]["ttft_proxy_s"] is not None]

        aggregate = {
            "avg_run_wall_time_s": round(statistics.mean(wall_times), 4) if wall_times else None,
            "max_run_wall_time_s": round(max(wall_times), 4) if wall_times else None,
            "min_run_wall_time_s": round(min(wall_times), 4) if wall_times else None,
            "avg_gen_tokens_per_s": round(statistics.mean(gen_tps), 2) if gen_tps else None,
            "avg_ttft_proxy_s": round(statistics.mean(ttfts), 4) if ttfts else None,
        }
        emit("   " + color(
            f"batch={fmt_s(round(batch_finished - batch_started, 4))} | avg_run={fmt_s(aggregate['avg_run_wall_time_s'])} | "
            f"max_run={fmt_s(aggregate['max_run_wall_time_s'])} | avg_out_tps={fmt_num(aggregate['avg_gen_tokens_per_s'])} | "
            f"avg_ttft≈{fmt_s(aggregate['avg_ttft_proxy_s'])}",
            C.WHITE,
        ))
        self.report["tests"].append({
            "name": name,
            "kind": "concurrent",
            "model": model,
            "prompt_name": prompt_name,
            "prompt_chars": len(prompt),
            "concurrency": concurrency,
            "keep_alive": keep_alive,
            "started_at": now_iso(),
            "batch_wall_time_s": round(batch_finished - batch_started, 4),
            "runs": sorted(runs, key=lambda x: x["index"]),
            "aggregate": aggregate,
        })

    def run_mixed_residency(self) -> None:
        if not self.worker_model:
            emit(color("Skipping mixed residency test.", C.YELLOW))
            return

        emit(color("→ mixed_residency_heavy_plus_worker", C.BOLD, C.YELLOW) + f"  main={color(self.primary_model, C.WHITE)}  worker={color(self.worker_model, C.WHITE)}")
        self.client.preload(self.primary_model, keep_alive=self.keep_alive)
        if self.worker_model != self.primary_model:
            self.client.preload(self.worker_model, keep_alive=self.keep_alive)
        time.sleep(0.6)

        def call(model: str, prompt_name: str, slot: str) -> Dict[str, Any]:
            started = time.perf_counter()
            response = self.client.generate(model=model, prompt=PROMPTS[prompt_name](), keep_alive=self.keep_alive, options=self._options())
            finished = time.perf_counter()
            return {
                "slot": slot,
                "model": model,
                "prompt_name": prompt_name,
                "metrics": shape_metrics(response, started, finished),
                "response_preview": (response.get("response") or "")[:500],
                "raw_response": response,
            }

        batch_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(call, self.primary_model, "long", "main")
            f2 = ex.submit(call, self.worker_model, "worker", "worker")
            main_result = f1.result()
            worker_result = f2.result()
        batch_finished = time.perf_counter()

        emit("   " + color(
            f"batch={fmt_s(round(batch_finished - batch_started, 4))} | main_total={fmt_s(main_result['metrics']['total_duration_s'])} | "
            f"worker_total={fmt_s(worker_result['metrics']['total_duration_s'])} | main_load={fmt_s(main_result['metrics']['load_duration_s'])} | "
            f"worker_load={fmt_s(worker_result['metrics']['load_duration_s'])}",
            C.WHITE,
        ))
        self.report["tests"].append({
            "name": "mixed_residency_heavy_plus_worker",
            "kind": "mixed",
            "started_at": now_iso(),
            "batch_wall_time_s": round(batch_finished - batch_started, 4),
            "main": main_result,
            "worker": worker_result,
        })

    def finalize_summary(self) -> None:
        tests = self.report["tests"]
        single = {t["name"]: t for t in tests if t["kind"] == "single"}
        concurrent = [t for t in tests if t["kind"] == "concurrent"]
        mixed = next((t for t in tests if t["kind"] == "mixed"), None)

        summary: Dict[str, Any] = {}
        if "cold_short" in single and "warm_short" in single:
            cold = single["cold_short"]["metrics"]
            warm = single["warm_short"]["metrics"]
            if cold["total_duration_s"] and warm["total_duration_s"]:
                summary["cold_to_warm_total_ratio"] = round(cold["total_duration_s"] / warm["total_duration_s"], 2)
            summary["cold_minus_warm_load_s"] = round((cold.get("load_duration_s") or 0) - (warm.get("load_duration_s") or 0), 4)
        if "warm_long" in single and "warm_xlong" in single:
            wl = single["warm_long"]["metrics"]["total_duration_s"]
            wx = single["warm_xlong"]["metrics"]["total_duration_s"]
            if wl and wx:
                summary["long_to_xlong_total_ratio"] = round(wx / wl, 2)
        if concurrent:
            summary["concurrency"] = {
                t["name"]: {
                    "batch_wall_time_s": t["batch_wall_time_s"],
                    "avg_run_wall_time_s": t["aggregate"].get("avg_run_wall_time_s"),
                    "avg_gen_tokens_per_s": t["aggregate"].get("avg_gen_tokens_per_s"),
                    "avg_ttft_proxy_s": t["aggregate"].get("avg_ttft_proxy_s"),
                }
                for t in concurrent
            }
        if mixed:
            summary["mixed_residency"] = {
                "batch_wall_time_s": mixed["batch_wall_time_s"],
                "main_load_duration_s": mixed["main"]["metrics"].get("load_duration_s"),
                "worker_load_duration_s": mixed["worker"]["metrics"].get("load_duration_s"),
                "main_total_duration_s": mixed["main"]["metrics"].get("total_duration_s"),
                "worker_total_duration_s": mixed["worker"]["metrics"].get("total_duration_s"),
            }
        self.report["summary"] = summary
        self.report["finished_at"] = now_iso()

    def save_report(self) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"ollama_stress_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
        return path



def get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def grade(total_s: Optional[float], warm_ratio: Optional[float]) -> str:
    if total_s is None:
        return 'unknown'
    score = 0
    if total_s <= 6:
        score += 2
    elif total_s <= 12:
        score += 1
    if warm_ratio is not None:
        if warm_ratio >= 3:
            score += 2
        elif warm_ratio >= 1.5:
            score += 1
    grades = ['rough', 'usable', 'good', 'strong', 'excellent']
    return grades[min(score, len(grades) - 1)]


def build_human_summary(report: Dict[str, Any], report_path: Optional[Path] = None) -> str:
    tests = report.get('tests', [])
    by_name: Dict[str, Dict[str, Any]] = {}
    for t in tests:
        name = t.get('name')
        if isinstance(name, str):
            by_name[name] = t

    summary = report.get('summary', {}) or {}

    cold_short = by_name.get('cold_short', {})
    warm_short = by_name.get('warm_short', {})
    warm_medium = by_name.get('warm_medium', {})
    warm_long = by_name.get('warm_long', {})
    warm_xlong = by_name.get('warm_xlong', {})
    conc2 = by_name.get('concurrent_2_medium', {})
    conc3 = by_name.get('concurrent_3_medium', {})
    mixed = by_name.get('mixed_residency_heavy_plus_worker', {})

    cold_m = cold_short.get('metrics', {})
    warm_m = warm_short.get('metrics', {})
    medium_m = warm_medium.get('metrics', {})
    long_m = warm_long.get('metrics', {})
    xlong_m = warm_xlong.get('metrics', {})
    conc2_a = conc2.get('aggregate', {})
    conc3_a = conc3.get('aggregate', {})
    mixed_main = get_nested(mixed, 'main', 'metrics') or {}
    mixed_worker = get_nested(mixed, 'worker', 'metrics') or {}

    cold_total = cold_m.get('total_duration_s')
    warm_total = warm_m.get('total_duration_s')
    warm_ratio = None
    if cold_total and warm_total:
        warm_ratio = cold_total / warm_total

    long_drag = None
    if medium_m.get('total_duration_s') and long_m.get('total_duration_s'):
        long_drag = long_m['total_duration_s'] / medium_m['total_duration_s']

    xlong_drag = None
    if long_m.get('total_duration_s') and xlong_m.get('total_duration_s'):
        xlong_drag = xlong_m['total_duration_s'] / long_m['total_duration_s']

    conc_penalty = None
    if warm_m.get('total_duration_s') and conc2_a.get('avg_run_wall_time_s'):
        conc_penalty = conc2_a['avg_run_wall_time_s'] / warm_m['total_duration_s']

    lines: List[str] = []
    lines.append('OLLAMA STRESS TEST SUMMARY')
    lines.append('=' * 27)
    if report_path:
        lines.append(f'Report: {report_path}')
    lines.append(f'Primary model: {report.get("model", "-")}')
    lines.append(f'Worker model:  {report.get("worker_model", "-")}')
    lines.append(f'Keep alive:    {report.get("keep_alive", "-")}')
    lines.append('')

    lines.append('At a glance')
    lines.append('-' * 10)
    lines.append(f'- Warm short total: {fmt_s(warm_total)}')
    lines.append(f'- Cold short total: {fmt_s(cold_total)}')
    lines.append(f'- Cold vs warm speedup: {fmt_num(warm_ratio)}x')
    lines.append(f'- Warm short TTFT proxy: {fmt_s(warm_m.get("ttft_proxy_s"))}')
    lines.append(f'- Warm short output speed: {fmt_num(warm_m.get("gen_tokens_per_s"))} tok/s')
    lines.append(f'- Overall feel: {grade(warm_total, warm_ratio)}')
    lines.append('')

    lines.append('Load behavior')
    lines.append('-' * 13)
    load_saved = None
    if cold_m.get("load_duration_s") is not None and warm_m.get("load_duration_s") is not None:
        load_saved = (cold_m.get("load_duration_s") or 0) - (warm_m.get("load_duration_s") or 0)
    lines.append(f'- Cold load time: {fmt_s(cold_m.get("load_duration_s"))}')
    lines.append(f'- Warm load time: {fmt_s(warm_m.get("load_duration_s"))}')
    lines.append(f'- Load savings after warmup: {fmt_s(load_saved)}')
    lines.append('')

    lines.append('Context scaling')
    lines.append('-' * 15)
    lines.append(f'- Warm medium total: {fmt_s(medium_m.get("total_duration_s"))}')
    lines.append(f'- Warm long total:   {fmt_s(long_m.get("total_duration_s"))} ({fmt_num(long_drag)}x vs medium)')
    lines.append(f'- Warm xlong total:  {fmt_s(xlong_m.get("total_duration_s"))} ({fmt_num(xlong_drag)}x vs long)')
    lines.append('')

    lines.append('Concurrency')
    lines.append('-' * 11)
    lines.append(f'- 2-way medium batch: avg run {fmt_s(conc2_a.get("avg_run_wall_time_s"))}, max run {fmt_s(conc2_a.get("max_run_wall_time_s"))}, avg out {fmt_num(conc2_a.get("avg_gen_tokens_per_s"))} tok/s')
    lines.append(f'- 3-way medium batch: avg run {fmt_s(conc3_a.get("avg_run_wall_time_s"))}, max run {fmt_s(conc3_a.get("max_run_wall_time_s"))}, avg out {fmt_num(conc3_a.get("avg_gen_tokens_per_s"))} tok/s')
    lines.append(f'- 2-way penalty vs warm short: {fmt_num(conc_penalty)}x')
    lines.append('')

    if mixed:
        lines.append('Mixed residency')
        lines.append('-' * 14)
        lines.append(f'- Main model total:   {fmt_s(mixed_main.get("total_duration_s"))} | load {fmt_s(mixed_main.get("load_duration_s"))}')
        lines.append(f'- Worker model total: {fmt_s(mixed_worker.get("total_duration_s"))} | load {fmt_s(mixed_worker.get("load_duration_s"))}')
        lines.append(f'- Batch wall time:    {fmt_s(mixed.get("batch_wall_time_s"))}')
        lines.append('')

    verdicts: List[str] = []
    if warm_total is not None:
        if warm_total <= 8:
            verdicts.append('Good warm latency for main-orchestrator use.')
        elif warm_total <= 15:
            verdicts.append('Usable as an orchestrator, but not snappy.')
        else:
            verdicts.append('Warm latency is heavy; keep this model for premium orchestration only.')

    if warm_ratio is not None:
        if warm_ratio >= 3:
            verdicts.append('Cold start is a major tax. Keep the model warm whenever possible.')
        elif warm_ratio >= 1.5:
            verdicts.append('Warmup helps noticeably, but cold starts are not catastrophic.')

    if conc2_a.get('avg_run_wall_time_s') and warm_total:
        ratio = conc2_a['avg_run_wall_time_s'] / warm_total
        if ratio >= 2.5:
            verdicts.append('Concurrency degrades hard. Keep worker count low.')
        elif ratio >= 1.5:
            verdicts.append('Concurrency is workable at 2-way load, but do not get greedy.')
        else:
            verdicts.append('2-way concurrency looks healthy.')

    if mixed:
        main_load = mixed_main.get('load_duration_s') or 0
        worker_load = mixed_worker.get('load_duration_s') or 0
        if main_load > 1 or worker_load > 1:
            verdicts.append('Mixed residency appears to trigger extra loading. Watch for VRAM churn.')
        else:
            verdicts.append('Mixed residency looks stable for this pairing.')

    lines.append('Bottom line')
    lines.append('-' * 11)
    if verdicts:
        lines.extend(f'- {v}' for v in verdicts)
    else:
        lines.append('- Not enough test data to form a useful verdict.')

    if summary:
        lines.append('')
        lines.append('Report summary fields')
        lines.append('-' * 21)
        for key, value in summary.items():
            if isinstance(value, float):
                rendered = fmt_num(value, 2)
            else:
                rendered = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            lines.append(f'- {key}: {rendered}')

    return '\n'.join(lines)


def write_summary_file(report_path: Path, text: str) -> Path:
    summary_path = report_path.with_name(report_path.stem + '_summary.txt')
    summary_path.write_text(text, encoding='utf-8')
    return summary_path


def print_human_summary(text: str) -> None:
    rule('Human Summary')
    for line in text.splitlines():
        if line == 'OLLAMA STRESS TEST SUMMARY':
            emit(color(line, C.CYAN, C.BOLD))
        elif line in {'At a glance', 'Load behavior', 'Context scaling', 'Concurrency', 'Mixed residency', 'Bottom line', 'Report summary fields'}:
            emit(color(line, C.YELLOW, C.BOLD))
        elif line.startswith('- '):
            emit(color(line, C.WHITE))
        elif line.startswith('=') or set(line) == {'-'}:
            emit(color(line, C.BRIGHT_BLACK))
        else:
            emit(line)


def prompt_view_summary() -> bool:
    raw = input(color('Stress test complete, view summary? [Y/n]: ', C.BOLD, C.GREEN)).strip().lower()
    return raw in ('', 'y', 'yes')

def print_single_test_summary(report: Dict[str, Any]) -> None:
    rule("Single Test Summary")
    for t in report["tests"]:
        if t["kind"] != "single":
            continue
        emit(color(f"{t['name']}", C.CYAN, C.BOLD) + ": " + summarize_metrics(t["metrics"]))


def main() -> int:
    try:
        models = list_models()
    except Exception as e:
        emit(color(f"Startup error: {e}", C.RED, C.BOLD))
        return 1

    primary = choose_model(models, "A) Please select primary model:")
    worker = choose_model(models, "B) Please select worker model:", allow_skip=True)
    keep_alive = prompt_keep_alive(DEFAULT_KEEP_ALIVE)

    panel("Ollama Stress Test", [
        f"Base URL: {BASE_URL}",
        f"Primary model: {primary}",
        f"Worker model: {worker or '-'}",
        f"Keep-alive: {keep_alive}",
        f"Output dir: {OUTPUT_DIR}",
    ])

    client = OllamaClient(BASE_URL)
    tester = StressTester(client, primary, worker, keep_alive)

    rule("Model Inspection")
    tester.report["environment"]["primary_model"] = tester.inspect_model(primary)
    if worker:
        tester.report["environment"]["worker_model"] = tester.inspect_model(worker)

    primary_info = tester.report["environment"]["primary_model"]
    if "error" in primary_info:
        emit(color(f"Could not inspect primary model: {primary_info['error']}", C.RED))
    else:
        details = primary_info.get("details") or {}
        emit(
            "Primary model details: "
            + color(f"family={details.get('family', '-')}", C.WHITE)
            + " | "
            + color(f"parameter_size={details.get('parameter_size', '-')}", C.WHITE)
            + " | "
            + color(f"quantization={details.get('quantization_level', '-')}", C.WHITE)
        )

    rule("Single-Request Tests")
    tester.run_single("cold_short", primary, "short", preload=False, force_cold=True, keep_alive=0)
    tester.run_single("warm_short", primary, "short", preload=True, force_cold=False, keep_alive=keep_alive)
    tester.run_single("warm_medium", primary, "medium", preload=True, force_cold=False, keep_alive=keep_alive)
    tester.run_single("warm_long", primary, "long", preload=True, force_cold=False, keep_alive=keep_alive)
    tester.run_single("warm_xlong", primary, "xlong", preload=True, force_cold=False, keep_alive=keep_alive)

    rule("Concurrency Tests")
    tester.run_concurrent("concurrent_2_medium", primary, "medium", 2, keep_alive=keep_alive)
    tester.run_concurrent("concurrent_3_medium", primary, "medium", 3, keep_alive=keep_alive)
    tester.run_concurrent("concurrent_2_long", primary, "long", 2, keep_alive=keep_alive)

    rule("Mixed Residency Test")
    tester.run_mixed_residency()

    rule("Cleanup")
    try:
        client.unload(primary)
        if worker and worker != primary:
            client.unload(worker)
        emit(color("Models unloaded with keep_alive=0.", C.GREEN))
    except Exception as e:
        emit(color(f"Cleanup note: {e}", C.YELLOW))

    tester.finalize_summary()
    print_single_test_summary(tester.report)
    report_path = tester.save_report()
    summary_text = build_human_summary(tester.report, report_path)
    summary_path = write_summary_file(report_path, summary_text)

    rule("Summary")
    emit(json.dumps(tester.report.get("summary", {}), indent=2))
    emit(color(f"Report written: {report_path}", C.BOLD, C.BRIGHT_GREEN))
    emit(color(f"Summary written: {summary_path}", C.BOLD, C.BRIGHT_GREEN))

    if prompt_view_summary():
        print_human_summary(summary_text)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        emit(color("\nInterrupted by user.", C.YELLOW, C.BOLD))
        raise SystemExit(130)
