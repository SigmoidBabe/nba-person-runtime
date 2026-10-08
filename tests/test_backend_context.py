"""Exercise backend context ownership without a GPU or inference dependencies."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class Context:
    def __init__(self):
        self.state = local()

    def stack(self):
        if not hasattr(self.state, "stack"):
            self.state.stack = []
        return self.state.stack

    def push(self):
        self.stack().append(self)

    def pop(self):
        self.stack().pop()


class BackendContextTests(unittest.TestCase):
    def setUp(self):
        self.context = Context()
        self.detector = Mock()
        self.reid = Mock()
        self.detector_factory = Mock(side_effect=self.make_detector)
        self.reid_factory = Mock(side_effect=self.make_reid)
        cuda = ModuleType("pycuda")
        cuda.autoinit = ModuleType("pycuda.autoinit")
        cuda.autoinit.context = self.context
        tracker = SimpleNamespace(BoTSORTConfig=Mock(), YOLOv8BoTSORT=Mock(), xyxy_polygon=Mock())
        self.modules = {
            "numpy": ModuleType("numpy"),
            "yaml": ModuleType("yaml"),
            "person_inference": SimpleNamespace(Detector=self.detector_factory),
            "osnet_reid_trt": SimpleNamespace(ReIDTRT=self.reid_factory),
            "cv_people.botsort_reid.botsort_tracker": tracker,
            "pycuda": cuda,
            "pycuda.autoinit": cuda.autoinit,
        }
        self.module_patch = patch.dict("sys.modules", self.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        path = Path(__file__).resolve().parents[1] / "src/cv_people/backend.py"
        spec = importlib.util.spec_from_file_location("cv_people.backend", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.backend_type = module.PeopleBackend

    def assert_active(self):
        self.assertIs(self.context.stack()[-1], self.context)

    def make_detector(self, path):
        if path.endswith(".trt"):
            self.assert_active()
        return self.detector

    def make_reid(self, path):
        self.assert_active()
        return self.reid

    def backend(self, person="person.trt", reid=None):
        backend = self.backend_type({"model": {"person": person, "reid": reid}})
        backend.convert_detection_result = lambda result: result
        return backend

    def test_detection_on_another_thread_restores_previous_context(self):
        backend = self.backend()
        self.assertEqual(self.context.stack(), [])

        def detect(image, **kwargs):
            self.assert_active()
            return "detections"

        self.detector.detect_n_seg.side_effect = detect

        def worker():
            previous = object()
            self.context.stack().append(previous)
            self.assertEqual(backend.detect("image", 0.4), "detections")
            self.assertEqual(self.context.stack(), [previous])

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(worker).result()

    def test_context_is_popped_after_inference_failure(self):
        backend = self.backend()

        def fail(*args, **kwargs):
            self.assert_active()
            raise RuntimeError("inference failed")

        self.detector.detect_n_seg.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            backend.detect("image", 0.4)
        self.assertEqual(self.context.stack(), [])

    def test_context_is_popped_after_constructor_failure(self):
        self.detector_factory.side_effect = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            self.backend()
        self.assertEqual(self.context.stack(), [])

    def test_reid_fallback_runs_with_context_active(self):
        backend = self.backend(reid="reid.engine")
        backend.to_feature_numpy = lambda feature: feature

        def batch(crops):
            self.assert_active()
            raise RuntimeError("batch failed")

        def single(crop):
            self.assert_active()
            return "feature"

        self.reid.extract_features.side_effect = batch
        self.reid.extract_feature.side_effect = single
        with ThreadPoolExecutor(max_workers=1) as executor:
            self.assertEqual(executor.submit(backend.extract_reid_batch, ["crop"]).result(), ["feature"])
        self.assertEqual(self.context.stack(), [])

    def test_non_trt_detector_does_not_import_cuda(self):
        with patch.dict("sys.modules", {"pycuda": None, "pycuda.autoinit": None}):
            backend = self.backend(person="person.onnx")
            backend.detect("image", 0.4)
            backend.close()
        self.assertIsNone(backend._cuda_context)

    def test_cleanup_activates_context_for_tracker_release(self):
        backend = self.backend()
        contexts_on_release = []
        context = self.context

        class Tracker:
            def __del__(self):
                contexts_on_release.append(list(context.stack()))

        backend.trackers_by_service_key["session"] = Tracker()
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(backend.close).result()
        self.assertEqual(contexts_on_release, [[self.context]])
        self.assertIsNone(backend.person_detector)
        self.assertIsNone(backend.person_reid)
        self.assertEqual(self.context.stack(), [])


if __name__ == "__main__":
    unittest.main()
