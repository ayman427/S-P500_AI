import gzip
import hashlib
import unittest
from unittest.mock import Mock, patch

from sp500_ml import cloud_data


class CloudDataTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.object(cloud_data, "_cache", cloud_data.OrderedDict())
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.size = patch.object(cloud_data, "_cache_size", 0)
        self.size.start()
        self.addCleanup(self.size.stop)

    def metadata(self, content):
        packed = gzip.compress(content, mtime=0)
        return packed, {"contentSha256": hashlib.sha256(content).hexdigest(),
                        "sha256": hashlib.sha256(packed).hexdigest(), "size": len(content), "url": "https://storage.example/file"}

    def test_read_verifies_bytes_and_in_memory_cache_uses_content_hash(self):
        content = b"Date,Close\n2026-10-14,100\n"
        packed, meta = self.metadata(content)
        with patch.object(cloud_data, "metadata", return_value=meta), patch.object(cloud_data.requests, "get", return_value=Mock(content=packed)) as get:
            self.assertEqual(cloud_data.read_bytes("prices/AAA.csv"), content)
            self.assertEqual(cloud_data.read_bytes("prices/AAA.csv"), content)
            get.assert_called_once()

    def test_corrupted_download_is_not_cached(self):
        packed, meta = self.metadata(b"valid")
        with patch.object(cloud_data, "metadata", return_value=meta), patch.object(cloud_data.requests, "get", return_value=Mock(content=packed + b"corruption")):
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                cloud_data.read_bytes("prices/AAA.csv")
            self.assertFalse(cloud_data._cache)

    def test_missing_cloud_data_never_reads_local_disk(self):
        with patch.object(cloud_data, "metadata", return_value=None), patch("builtins.open", side_effect=AssertionError("Disk access")):
            with self.assertRaises(FileNotFoundError):
                cloud_data.read_bytes("prices/AAA.csv")

    def test_write_commits_uploaded_checksum_with_expected_version(self):
        with patch.object(cloud_data, "metadata", return_value={"version": 4, "contentSha256": "old"}), patch.object(cloud_data, "rpc", side_effect=["https://upload.example", {"version": 5}]) as rpc, patch.object(cloud_data.requests, "post", return_value=Mock(json=lambda: {"storageId": "storage"})):
            result = cloud_data.write_bytes("news/AAA.json", b"{}")
            self.assertEqual(result["version"], 5)
            args = rpc.call_args.args[2]
            self.assertEqual(args["expectedVersion"], 4)
            self.assertEqual(args["contentSha256"], hashlib.sha256(b"{}").hexdigest())

    def test_list_paginates_all_records(self):
        with patch.object(cloud_data, "rpc", side_effect=[{"page": [{"key": "a"}], "isDone": False, "continueCursor": "next"}, {"page": [{"key": "b"}], "isDone": True}]) as rpc:
            self.assertEqual(len(cloud_data.list_datasets("news/")), 2)
            self.assertEqual(rpc.call_args.args[2]["cursor"], "next")

    def test_csv_and_json_writes_use_cloud_without_local_files(self):
        import pandas as pd
        with patch.object(cloud_data, "write_bytes") as write:
            cloud_data.write_csv(pd.DataFrame({"Close": [100]}, index=pd.DatetimeIndex(["2026-10-14"], name="Date")), cloud_data.RemotePath("prices/AAA.csv"))
            self.assertIn(b"2026-10-14", write.call_args.args[1])
            cloud_data.RemotePath("news/AAA.json").write_text("{}")
            self.assertEqual(write.call_args.args, ("news/AAA.json", b"{}"))

    def test_model_generation_is_uploaded_and_manifest_is_published_last(self):
        import json
        import pandas as pd
        from contextlib import nullcontext
        from sp500_ml.artifacts import publish_generation, resolve_generation
        stored = {}
        order = []
        def write(key, payload, modified_at=None):
            order.append(key)
            stored[key] = payload
        with patch.object(cloud_data, "write_bytes", side_effect=write), patch.object(cloud_data, "read_bytes", side_effect=lambda key: stored[key]), patch.object(cloud_data, "metadata", side_effect=lambda key: {} if key in stored else None), patch("sp500_ml.artifacts.exclusive_cloud_operation", return_value=nullcontext()):
            model, metrics, holdout, trades = publish_generation(cloud_data.RemotePath("models"), "AAA_ridge_open_to_close", {"ticker": "AAA"}, {}, {}, pd.DataFrame({"return": [.01]}))
            self.assertTrue(order[-1].endswith("_manifest.json"))
            import joblib
            from io import BytesIO
            bundle = joblib.load(BytesIO(model.read_bytes()))
            self.assertEqual(bundle["run_id"], json.loads(metrics.read_text())["run_id"])
            self.assertEqual(resolve_generation(cloud_data.RemotePath("models"), "AAA_ridge_open_to_close")[0], model)
            stored[model.key] = b"corrupt"
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                resolve_generation(cloud_data.RemotePath("models"), "AAA_ridge_open_to_close")

    def test_cloud_lock_is_released_when_operation_fails(self):
        with patch.object(cloud_data, "rpc", return_value=True) as rpc:
            with self.assertRaisesRegex(ValueError, "failure"):
                with cloud_data.exclusive_cloud_operation("paper_trades/AAA.lock"):
                    self.assertIsNotNone(cloud_data._lease.get())
                    raise ValueError("failure")
            self.assertIsNone(cloud_data._lease.get())
            self.assertEqual(rpc.call_args_list[0].args[1], "acquireLock")
            self.assertEqual(rpc.call_args_list[-1].args[1], "releaseLock")
