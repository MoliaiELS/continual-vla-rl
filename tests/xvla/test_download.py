"""HTTP resume failure cases; no model download or GPU execution."""
import hashlib
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "tools/xvla/download.py"
SPEC = importlib.util.spec_from_file_location("xvla_fixed_downloader", SOURCE)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)
PAYLOAD = bytes(range(256)) * 128


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        start = int(self.headers["Range"].split("=")[1].split("-")[0])
        self.server.offsets.append(start)
        if self.server.ignore_range:
            self.send_response(200)
            body = PAYLOAD
        else:
            self.send_response(206)
            value = start + int(self.server.bad_range)
            self.send_header("Content-Range", f"bytes {value}-{len(PAYLOAD)-1}/{len(PAYLOAD)}")
            body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.ignore_range = False
        self.server.bad_range = False
        self.server.offsets = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.entry = {"rfilename": "model.safetensors", "size": len(PAYLOAD),
                      "lfs": {"size": len(PAYLOAD), "sha256": hashlib.sha256(PAYLOAD).hexdigest()}}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def fetch(self):
        endpoint = "http://127.0.0.1:" + str(self.server.server_port)
        return download.download_file("org/model", "a" * 40, self.entry, self.root, endpoint)

    def test_resume_preserves_correct_prefix_and_verifies_full_hash(self):
        (self.root / "model.safetensors.http.part").write_bytes(PAYLOAD[:4000])
        result = self.fetch()
        self.assertEqual(self.server.offsets, [4000])
        self.assertEqual((self.root / "model.safetensors").read_bytes(), PAYLOAD)
        self.assertEqual(result["sha256"], self.entry["lfs"]["sha256"])

    def test_corrupt_xet_prefix_restarts_zero_and_keeps_old_cache(self):
        cached = self.root / ".cache" / ("old." + self.entry["lfs"]["sha256"] + ".incomplete")
        cached.parent.mkdir()
        cached.write_bytes(b"bad!" * 1000)
        self.fetch()
        self.assertEqual(self.server.offsets, [4000, 0])
        self.assertEqual(cached.read_bytes(), b"bad!" * 1000)
        self.assertEqual((self.root / "model.safetensors").read_bytes(), PAYLOAD)

    def test_server_ignoring_range_restarts_using_complete_body(self):
        (self.root / "model.safetensors.http.part").write_bytes(PAYLOAD[:4000])
        self.server.ignore_range = True
        self.fetch()
        self.assertEqual((self.root / "model.safetensors").read_bytes(), PAYLOAD)

    def test_mismatched_content_range_never_publishes_asset(self):
        self.server.bad_range = True
        with self.assertRaises(ValueError):
            self.fetch()
        self.assertFalse((self.root / "model.safetensors").exists())

    def test_non_lfs_assets_require_git_blob_hash(self):
        target = self.root / "config.json"
        target.write_bytes(b'{"safe":true}')
        body = target.read_bytes()
        entry = {"rfilename": "config.json", "size": len(body), "lfs": None,
                 "blob_id": hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()}
        self.assertIsNotNone(download.verify_file(target, entry))
        target.write_bytes(b'{"safe":fals}')
        self.assertIsNone(download.verify_file(target, entry))


if __name__ == "__main__":
    unittest.main()
