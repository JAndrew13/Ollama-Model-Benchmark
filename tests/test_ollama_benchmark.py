import json
import unittest
from unittest.mock import patch

import ollama_benchmark as ob


class ParsingTests(unittest.TestCase):
    def test_parse_modelfile_options(self):
        modelfile = """
        FROM base
        PARAMETER temperature 0.7
        PARAMETER num_ctx 8192
        PARAMETER temperature 0.8
        """
        self.assertEqual(ob.parse_modelfile_options(modelfile), ["num_ctx", "temperature"])

    def test_parse_model_parameters(self):
        text = "temperature 0.8\nnum_ctx 8192\ntemperature 0.2"
        self.assertEqual(ob.parse_model_parameters(text), {"temperature": "0.8", "num_ctx": "8192"})


class InstalledModelTests(unittest.TestCase):
    def test_get_installed_models_json(self):
        def runner(cmd):
            self.assertEqual(cmd, ["ollama", "list", "--json"])
            return json.dumps([{"name": "model-a"}])

        self.assertEqual(ob.get_installed_models(command_runner=runner), [{"name": "model-a"}])

    def test_get_installed_models_fallback(self):
        calls = []

        def runner(cmd):
            calls.append(cmd)
            if cmd == ["ollama", "list", "--json"]:
                raise RuntimeError("unsupported")
            return "NAME SIZE\nmodel-a:latest 2GB\n"

        self.assertEqual(ob.get_installed_models(command_runner=runner), [{"name": "model-a:latest"}])
        self.assertEqual(calls[1], ["ollama", "list"])


class MetadataAndScoringTests(unittest.TestCase):
    def test_get_model_show_json_via_api(self):
        payload = {"details": {"family": "qwen2"}, "capabilities": ["completion"]}

        def fetcher(model):
            self.assertEqual(model, "model-a")
            return payload

        self.assertEqual(ob.get_model_show_json("model-a", show_fetcher=fetcher), payload)

    def test_get_model_show_json_handles_total_failure(self):
        def fetcher(_model):
            raise RuntimeError("show failed")

        self.assertEqual(ob.get_model_show_json("broken", show_fetcher=fetcher), {})

    def test_get_model_metadata(self):
        payload = {
            "model_info": {
                "general.architecture": "llama",
                "general.parameter_count": 7000000000,
                "llama.embedding_length": 4096,
                "llama.context_length": 8192,
                "general.file_type": "Q4_K_M",
            },
            "details": {},
            "capabilities": ["completion", "tools", "vision", "embedding"],
            "parameters": "temperature 0.6\nnum_ctx 8192",
            "modelfile": "PARAMETER temperature 0.6\nPARAMETER num_ctx 8192",
        }

        meta = ob.get_model_metadata("reasoning-model", show_fetcher=lambda _model: payload)
        self.assertEqual(meta["architecture"], "llama")
        self.assertEqual(meta["parameters"], 7.0)
        self.assertTrue(meta["acceptsTools"])
        self.assertTrue(meta["reasoning"])
        self.assertIn("image", meta["inputTypes"])
        self.assertIn("vector", meta["outputTypes"])

    def test_get_model_metadata_prefers_details_quantization_level(self):
        payload = {
            "model_info": {
                "general.architecture": "qwen2",
                "general.parameter_count": 14800000000,
                "general.file_type": "15",
            },
            "details": {"quantization_level": "Q4_K_M"},
            "capabilities": ["completion"],
        }

        meta = ob.get_model_metadata("qwen2.5-coder:14b", show_fetcher=lambda _model: payload)
        self.assertEqual(meta["quantization"], "Q4_K_M")

    def test_efficiency_and_runtime_profiles(self):
        efficiency = ob.get_efficiency_profile(40, 0.2, 1.0, 8, 8192)
        self.assertEqual(efficiency["tokensPerSecondPerBParameter"], 5.0)
        self.assertEqual(efficiency["tokensPerSecondPerGBVram"], 5.0)

        runtime = ob.get_optimal_runtime_settings(65536, 40, 0.2, {"temperature": "0.3"})
        self.assertEqual(runtime["profile"], "low-latency")
        self.assertEqual(runtime["recommendedOptions"]["num_predict"], 1024)

    def test_workload_recommendations_default_and_extended(self):
        default = ob.get_workload_recommendations(1, 2, 4096, False, [], False)
        self.assertEqual(default, ["general-purpose text generation"])

        rich = ob.get_workload_recommendations(30, 0.5, 131072, True, ["vision"], True)
        self.assertIn("interactive chat and coding assistants", rich)
        self.assertIn("multimodal image+text analysis", rich)


class OutputPathTests(unittest.TestCase):
    def test_build_default_output_path_single_model(self):
        dt = ob.datetime(2024, 1, 2, 3, 4, 5)
        path = ob.build_default_output_path("qwen2.5-coder:14b", now=dt)
        self.assertEqual(path.as_posix(), "outputs/Ollama_qwen2.5-coder_14b_benchmark_20240102_030405.json")

    def test_build_default_output_path_multi_model(self):
        dt = ob.datetime(2024, 1, 2, 3, 4, 5)
        path = ob.build_default_output_path(None, now=dt)
        self.assertEqual(path.as_posix(), "outputs/Ollama_multi_benchmark_20240102_030405.json")



class BenchmarkFlowTests(unittest.TestCase):
    def test_context_window_stops_on_failure(self):
        attempts = []

        def generator(model, prompt, options):
            attempts.append(options["num_ctx"])
            if options["num_ctx"] > 16384:
                raise RuntimeError("too large")
            return {"response": "ok"}

        ctx = ob.test_context_window("test", generator=generator)
        self.assertEqual(ctx, 16384)
        self.assertEqual(attempts[:3], [8192, 16384, 32768])

    def test_benchmark_generation(self):
        snapshots = iter([
            {"util": 10, "mem": 1024},
            {"util": 40, "mem": 2048},
        ])
        cpus = iter([10.1, 15.2])

        def generator(model, prompt, options):
            return {
                "load_duration": 1_000_000_000,
                "prompt_eval_duration": 500_000_000,
                "eval_duration": 2_000_000_000,
                "eval_count": 100,
                "prompt_eval_count": 12,
            }

        result = ob.benchmark_generation(
            "model",
            generator=generator,
            gpu_snapshotter=lambda: next(snapshots),
            cpu_snapshotter=lambda: next(cpus),
        )

        self.assertEqual(result["tokensPerSecond"], 50.0)
        self.assertEqual(result["firstToken"], 1.5)
        self.assertEqual(result["gpuMem"], 2048)
        self.assertEqual(result["cpuUtil"], 15.2)

    @patch("ollama_benchmark.benchmark_generation")
    @patch("ollama_benchmark.test_context_window")
    @patch("ollama_benchmark.get_model_metadata")
    @patch("ollama_benchmark.get_installed_models")
    def test_run_benchmark_single_model(self, mock_models, mock_meta, mock_ctx, mock_bench):
        mock_models.return_value = [{"name": "m1"}, {"name": "m2"}]
        mock_meta.return_value = {
            "architecture": "llama",
            "parameters": 7.0,
            "embeddingLength": 0,
            "contextLength": 8192,
            "quantization": "Q4",
            "capabilities": ["completion"],
            "inputTypes": ["text"],
            "outputTypes": ["text"],
            "acceptsTools": False,
            "reasoning": False,
            "options": {"temperature": "0.7"},
            "optionsAvailable": ["temperature"],
        }
        mock_ctx.return_value = 16384
        mock_bench.return_value = {
            "loadTime": 1.0,
            "firstToken": 0.2,
            "tokensPerSecond": 40,
            "generatedTokens": 200,
            "promptTokens": 10,
            "evalDurationSec": 5,
            "gpuMem": 0,
            "gpuUtil": 0,
            "cpuUtil": 1,
        }

        results = ob.run_benchmark(model_id="m1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "m1")
        self.assertEqual(results[0]["contextWindow"], 8192)


if __name__ == "__main__":
    unittest.main()
