import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from qwen_vl.data.rgb_frame_cache import ExactRGBFrameCache, RGBFrameCacheError


class ExactRGBFrameCacheTest(unittest.TestCase):
    def test_roundtrip_and_exact_frame_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "scene.mp4"
            video.write_bytes(b"fixture video")
            cache = ExactRGBFrameCache(str(root / "rgb"))
            calls = []

            def decode():
                calls.append(1)
                return [Image.new("RGB", (3, 2), (frame_id, 4, 5)) for frame_id in (1, 7)]

            first, hit = cache.load_or_create(video, [1, 7], decode)
            self.assertFalse(hit)
            second, hit = cache.load_or_create(video, [1, 7], decode)
            self.assertTrue(hit)
            self.assertEqual(len(calls), 1)
            self.assertEqual([image.tobytes() for image in first], [image.tobytes() for image in second])
            self.assertEqual([image.getpixel((0, 0))[0] for image in second], [1, 7])

            # A different selection gets a different entry and cannot borrow these pixels.
            _, hit = cache.load_or_create(video, [1, 8], lambda: [
                Image.new("RGB", (3, 2), (1, 4, 5)),
                Image.new("RGB", (3, 2), (8, 4, 5)),
            ])
            self.assertFalse(hit)

    def test_cross_node_device_number_does_not_invalidate_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "scene.mp4"
            video.write_bytes(b"fixture video")
            cache = ExactRGBFrameCache(str(root / "rgb"))
            cache.load_or_create(video, [1], lambda: [Image.new("RGB", (2, 2), (1, 2, 3))])
            path = cache._path(video, [1])
            with np.load(path, allow_pickle=False) as stored:
                rgb = stored["rgb"]
                metadata = json.loads(stored["metadata"].tobytes().decode("utf-8"))
            metadata["source"]["device"] = 999999
            with path.open("wb") as output:
                np.savez_compressed(
                    output,
                    rgb=rgb,
                    metadata=np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8),
                )
            images, hit = cache.load_or_create(video, [1], lambda: [])
            self.assertTrue(hit)
            self.assertEqual(images[0].getpixel((0, 0)), (1, 2, 3))

    def test_corrupt_or_stale_cache_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "scene.mp4"
            video.write_bytes(b"original")
            cache = ExactRGBFrameCache(str(root / "rgb"))
            decode = lambda: [Image.new("RGB", (2, 2), (3, 4, 5))]
            cache.load_or_create(video, [3], decode)
            video.write_bytes(b"changed")
            with self.assertRaisesRegex(RGBFrameCacheError, "source mismatch"):
                cache.load_or_create(video, [3], decode)

            cache._path(video, [3]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(RGBFrameCacheError, "Cannot read"):
                cache.load_or_create(video, [3], decode)

    def test_parallel_cache_miss_decodes_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "scene.mp4"
            video.write_bytes(b"fixture video")
            cache = ExactRGBFrameCache(str(root / "rgb"))
            calls = []

            def decode():
                calls.append(1)
                time.sleep(0.1)
                return [Image.new("RGB", (2, 2), (7, 8, 9))]

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: cache.load_or_create(video, [7], decode), range(2)))
            self.assertEqual(len(calls), 1)
            self.assertEqual(sorted(hit for _, hit in results), [False, True])
            self.assertEqual([images[0].getpixel((0, 0)) for images, _ in results], [(7, 8, 9)] * 2)


if __name__ == "__main__":
    unittest.main()
