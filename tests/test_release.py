"""CPU-only checks; no downloaded weights, datasets or CUDA dependencies."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock
import types
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpr_sam3.runtime import RuntimePaths, load_profile, normalized_hash


def research_method(relative, class_name, method_name):
    """Load an unchanged tensor helper without importing the CUDA/MMseg stack."""
    import torch
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(relative), "exec"), namespace)
    return namespace[method_name]


class ReleaseTests(unittest.TestCase):
    def test_dependency_pins_match_recorded_framework_versions(self):
        from packaging.requirements import Requirement

        requirements = [
            Requirement(line) for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        by_name = {item.name: item for item in requirements}
        self.assertEqual(len(by_name), len(requirements))
        expected = {"torch": "2.10.0", "torchvision": "0.25.0", "numpy": "1.26.4",
                    "mmcv": "2.1.0", "mmengine": "0.10.7", "mmsegmentation": "1.2.2"}
        for name, version in expected.items():
            self.assertEqual(str(by_name[name].specifier), "==" + version)
        self.assertIn("2.10.0+cu128", by_name["torch"].specifier)
        self.assertNotIn("2.0.0", by_name["numpy"].specifier)
        self.assertNotIn("mmcv-lite", by_name)
        self.assertIn("opencv-python", by_name)
        self.assertNotIn("opencv-python-headless", by_name)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("--index-url https://download.pytorch.org/whl/cu128", readme)

    def test_doctor_finds_bundled_sam3_without_importing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "upstream/sam3"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("raise RuntimeError('Must not import model code for discovery')\n")
            settings = dict(segearthov3_repo="upstream", dinov3_repo="dino", sam3_checkpoint="s.pt", dinov3_checkpoint="d.pt", sam3_bpe="bpe.gz")
            paths = root / "paths.json"
            paths.write_text(json.dumps(settings))
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/doctor.py"), "--paths", str(paths)],
                cwd=directory, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("OK package sam3 (SegEarth-OV3 checkout)", result.stdout)
            self.assertNotIn("Must not import model code", result.stdout + result.stderr)

    def test_documented_baseline_revision_matches_provenance(self):
        provenance = json.loads((ROOT / "provenance.json").read_text())
        commit = provenance["external_baseline"]["commit"]
        self.assertEqual(len(commit), 40)
        int(commit, 16)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"git -C third_party/SegEarth-OV-3 checkout {commit}", readme)

    def test_root_entry_points_work_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("demo.py", "eval.py", "build_prototypes.py"):
                result = subprocess.run(
                    [sys.executable, str(ROOT / name), "--help"],
                    cwd=directory, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--dataset", result.stdout)
                self.assertIn("--paths", result.stdout)

    def test_source_archive_includes_readme_figures_and_entry_points(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.zip"
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/make_archive.py"), "--output", str(output)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                for name in ("assets/framework.png", "assets/qualitative.png", "demo.py", "eval.py", "build_prototypes.py"):
                    self.assertIn("CPR-SAM3/" + name, names)
                self.assertNotIn("CPR-SAM3/paths.json", names)
                self.assertFalse(any("/.git/" in name for name in names))
                self.assertIsNone(archive.testzip())

    def test_profiles_and_prompts(self):
        expected = dict(vaihingen=6, potsdam=6, loveda=7, openearthmap=9)
        for name, count in expected.items():
            profile = load_profile(name)
            bank = json.loads((ROOT / "configs/prompt_banks" / profile["prompt_bank"]).read_text())
            self.assertEqual(len(bank["classes"]), count)
            for entry in bank["classes"]:
                self.assertEqual(len(entry["prompts"]), len(set(entry["prompts"])))
        with self.assertRaises(ValueError):
            load_profile("../private")

    def test_paths_are_relative_to_config_not_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = dict(segearthov3_repo="upstream", dinov3_repo="dino", sam3_checkpoint="s.pt", dinov3_checkpoint="d.pt", sam3_bpe="bpe.gz")
            path = root / "paths.json"
            path.write_text(json.dumps(settings))
            result = RuntimePaths.load(path)
            self.assertEqual(result.upstream, root / "upstream")
            self.assertEqual(len(result.problems()), 6)

    def test_hash_is_line_ending_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.py"
            path.write_bytes(b"a\r\nb\r\n")
            first = normalized_hash(path)
            path.write_bytes(b"a\nb\n")
            self.assertEqual(first, normalized_hash(path))

    def test_demo_does_not_load_labels(self):
        tree = ast.parse((ROOT / "tools/demo.py").read_text())
        constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertNotIn("gt_sem_seg", constants)
        self.assertNotIn("LoadAnnotations", constants)

    def test_core_imports_do_not_require_original_ours_namespace(self):
        for path in (ROOT / "cpr_sam3").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("ours."), str(path))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_public_model_import_and_result_selection_with_framework_stubs(self):
        import importlib
        import torch

        registry = types.ModuleType("mmseg.registry")

        class Registry:
            def register_module(self):
                return lambda cls: cls

        registry.MODELS = Registry()
        structures = types.ModuleType("mmengine.structures")
        structures.PixelData = lambda **kwargs: types.SimpleNamespace(**kwargs)
        baseline = types.ModuleType("segearthov3_segmentor")
        baseline.SegEarthOV3Segmentation = type("SegEarthOV3Segmentation", (torch.nn.Module,), {})
        replacements = {
            "mmseg": types.ModuleType("mmseg"), "mmseg.registry": registry,
            "mmengine": types.ModuleType("mmengine"), "mmengine.structures": structures,
            "segearthov3_segmentor": baseline,
        }
        before = set(sys.modules)
        try:
            with mock.patch.dict(sys.modules, replacements):
                module = importlib.import_module("cpr_sam3.model")
                self.assertTrue(callable(module.CPRSAM3Segmentor.calibration_observation))
                modes = module.OUTPUT_MODES
                index = modes.index("evidence_max_direct_vote")
                labels = torch.arange(len(modes)).view(-1, 1, 1).expand(-1, 2, 2)

                class Sample:
                    def __init__(self):
                        self.cetc_candidate_preds = types.SimpleNamespace(data=labels)
                        self.seg_logits = torch.zeros((1, 2, 2))

                    def set_data(self, values):
                        self.__dict__.update(values)

                    def __contains__(self, key):
                        return key in self.__dict__

                instance = object.__new__(module.CPRSAM3Segmentor)
                torch.nn.Module.__init__(instance)
                instance._result_index = index
                sample = Sample()
                with mock.patch.object(module.CrossExpertTrajectorySegmentor, "predict", return_value=[sample]):
                    result = instance.predict(None, [sample])[0]
                self.assertEqual(tuple(result.pred_sem_seg.data.shape), (1, 2, 2))
                self.assertTrue(result.pred_sem_seg.data.eq(index).all())
                self.assertFalse(hasattr(result, "seg_logits"))
        finally:
            for name in set(sys.modules) - before:
                if name == "cpr_sam3.model" or name.startswith("cpr_sam3.research"):
                    del sys.modules[name]

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_trajectory_requires_visual_improvement_and_respects_budget(self):
        import torch
        method = research_method("cpr_sam3/research/trajectory_cross_foundation_selector/segmentor.py", "TrajectoryCrossFoundationSegmentor", "_trajectory_candidate")
        instance = type("Instance", (), {"num_cls": 2})()
        base = torch.zeros((2, 2), dtype=torch.long)
        pool = torch.ones((2, 2, 2), dtype=torch.long)
        similarity = torch.stack((torch.zeros((2, 2)), torch.ones((2, 2))))
        support = torch.ones_like(similarity)
        result = method(instance, base, pool, similarity, support, 1, 0.0, 0.35, 0.25)
        self.assertEqual(int(result.sum()), 1)
        blocked = method(instance, base, pool, similarity, support, 3, 0.0, 0.35, 1.0)
        self.assertTrue(torch.equal(blocked, base))
        blocked = method(instance, base, pool, -similarity, support, 1, 0.0, 0.35, 1.0)
        self.assertTrue(torch.equal(blocked, base))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_evidence_switch_keeps_default_on_ties(self):
        import torch
        switch = research_method("cpr_sam3/research/cross_expert_trajectory_consensus/segmentor.py", "CrossExpertTrajectorySegmentor", "_switch")
        base = torch.tensor([[0, 0]])
        alternative = torch.tensor([[1, 1]])
        result = switch(base, alternative, torch.tensor([[0.0, 0.1]]), 0.0)
        self.assertEqual(result.tolist(), [[0, 1]])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_prototype_builder_handles_absent_classes(self):
        import torch
        from cpr_sam3.prototypes import clustered_prototypes, image_paths, image_prototypes
        logits = torch.stack((torch.ones((4, 4)), torch.zeros((4, 4))))
        features = torch.ones((3, 4, 4))
        records = image_prototypes(logits, features, 96)
        prototypes, valid, counts = clustered_prototypes(records, 2, 3, 3)
        self.assertEqual(tuple(prototypes.shape), (2, 3, 3))
        self.assertTrue(valid[0, 0])
        self.assertFalse(valid[1].any())
        self.assertEqual(counts.tolist(), [1, 0])
        with tempfile.TemporaryDirectory() as directory:
            for index in range(10):
                (Path(directory) / f"{index:02}.png").touch()
            paths = image_paths(directory, 3)
            self.assertEqual([p.name for p in paths], ["00.png", "04.png", "09.png"])


if __name__ == "__main__":
    unittest.main()
