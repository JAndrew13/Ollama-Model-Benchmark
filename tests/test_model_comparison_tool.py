import unittest

import model_comparison_tool as mct


class CompatibilityTests(unittest.TestCase):
    def test_score_local_compatibility(self):
        hardware = {"memory": {"gpuVramGB": 12, "systemRamGB": 32}}
        self.assertEqual(mct.score_local_compatibility(8, hardware), 1)
        self.assertEqual(mct.score_local_compatibility(12, hardware), 2)
        self.assertEqual(mct.score_local_compatibility(15, hardware), 3)
        self.assertEqual(mct.score_local_compatibility(30, hardware), 4)
        self.assertEqual(mct.score_local_compatibility(60, hardware), 5)

    def test_estimate_required_memory(self):
        self.assertGreater(mct.estimate_required_memory_gb(size_bytes=8 * 1024**3, parameter_billions=0, quantization=""), 9)
        self.assertGreater(mct.estimate_required_memory_gb(size_bytes=0, parameter_billions=7, quantization="Q4_K_M"), 3)


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.hardware = {"memory": {"gpuVramGB": 16, "systemRamGB": 64}}

    def test_ollama_to_entry(self):
        model = {
            "name": "qwen2.5:7b",
            "model": "qwen2.5:7b",
            "size": 4 * 1024**3,
            "details": {
                "family": "qwen2",
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
            },
        }
        entry = mct.ollama_to_entry(model, self.hardware)
        self.assertEqual(entry["provider"], "ollama")
        self.assertEqual(entry["family"], "qwen2")
        self.assertEqual(entry["compatibilityScore"], 1)

    def test_openrouter_to_entry(self):
        model = {
            "id": "z-ai/glm-4.7-flash",
            "name": "GLM 4.7 Flash 30B",
            "description": "A 30B-class SOTA model",
            "context_length": 203000,
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "quantization": "bf16",
            },
            "pricing": {"prompt": "0.00000006", "completion": "0.00000040"},
            "supported_parameters": ["tools"],
        }
        entry = mct.openrouter_to_entry(model, self.hardware)
        self.assertEqual(entry["provider"], "openrouter")
        self.assertTrue(entry["supportsToolUse"])
        self.assertEqual(entry["pricing"]["inputPerMTokensUSD"], 0.06)


if __name__ == "__main__":
    unittest.main()
