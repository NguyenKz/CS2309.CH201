from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.hybrid_editing import (
    Candidate,
    commit_candidate,
    crop_square,
    ensure_session,
    hybrid_composite,
    model_mask_to_roi,
    model_mask_to_original,
    parse_square_roi,
    paste_square,
    resolve_edit_mode,
    set_candidates,
    square_roi_from_mask,
    undo_session,
)


class HybridEditingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temp_dir.name) / "source.png"
        Image.new("RGB", (800, 600), (10, 20, 30)).save(self.image_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_mask_mapping_from_latent_canvas_to_landscape_image(self) -> None:
        mask = np.zeros((64, 64), np.float32)
        mask[16:48, 16:48] = 1
        meta = {
            "pad": (0, 64),
            "content_size": (512, 384),
            "orig_size": (800, 600),
        }
        mapped = model_mask_to_original(mask, meta)
        self.assertEqual(mapped.size, (800, 600))
        arr = np.asarray(mapped)
        self.assertGreater(arr[300, 400], 0)
        self.assertEqual(arr[10, 10], 0)

    def test_local_composite_preserves_pixels_outside_mask(self) -> None:
        master = Image.new("RGB", (8, 8), (10, 20, 30))
        edited = Image.new("RGB", (8, 8), (200, 210, 220))
        mask = Image.new("L", (8, 8), 0)
        mask.putpixel((4, 4), 255)
        result = hybrid_composite(
            master,
            edited,
            mask,
            mode="local",
            dilation=0,
            blur=0,
        )
        arr = np.asarray(result)
        self.assertTrue(np.array_equal(arr[0, 0], [10, 20, 30]))
        self.assertTrue(np.array_equal(arr[4, 4], [200, 210, 220]))

    def test_candidate_commit_and_undo(self) -> None:
        session = ensure_session(None, self.image_path, "source")
        candidate_image = Image.new("RGB", (800, 600), (100, 110, 120))
        candidate = Candidate(
            image=candidate_image,
            model_image=Image.new("RGB", (512, 512)),
            mask=Image.new("L", (800, 600), 255),
            clean_latent="latent",
            seed=123,
            mode="global",
            source_prompt="source",
            edit_prompt="edited",
        )
        set_candidates(session, [candidate])
        commit_candidate(session, 0)
        self.assertEqual(session.turn, 1)
        self.assertEqual(session.source_prompt, "edited")
        self.assertEqual(session.clean_latent, "latent")
        self.assertEqual(session.master.getpixel((0, 0)), (100, 110, 120))
        undo_session(session)
        self.assertEqual(session.turn, 0)
        self.assertEqual(session.source_prompt, "source")
        self.assertEqual(session.master.getpixel((0, 0)), (10, 20, 30))

    def test_new_upload_invalidates_session(self) -> None:
        session = ensure_session(None, self.image_path, "first")
        second_path = Path(self.temp_dir.name) / "second.png"
        Image.new("RGB", (320, 200), (1, 2, 3)).save(second_path)
        next_session = ensure_session(session, second_path, "second")
        self.assertIsNot(next_session, session)
        self.assertEqual(next_session.master.size, (320, 200))

    def test_auto_mode_switches_on_mask_coverage(self) -> None:
        self.assertEqual(resolve_edit_mode("auto", 0.1), "local")
        self.assertEqual(resolve_edit_mode("auto", 0.8), "global")
        self.assertEqual(resolve_edit_mode("local", 0.9), "local")

    def test_square_roi_crop_and_paste_preserve_full_image_size(self) -> None:
        roi = parse_square_roi('{"x": 100, "y": 50, "size": 200}', (800, 600))
        source = Image.open(self.image_path).convert("RGB")
        crop = crop_square(source, roi)
        self.assertEqual(crop.size, (200, 200))
        replacement = Image.new("RGB", (512, 512), (200, 0, 0))
        result = paste_square(source, replacement, roi)
        self.assertEqual(result.size, (800, 600))
        self.assertEqual(result.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(result.getpixel((150, 100)), (200, 0, 0))

    def test_square_roi_is_clamped_and_model_mask_is_placed(self) -> None:
        roi = parse_square_roi('{"x": 750, "y": 550, "size": 200}', (800, 600))
        self.assertEqual((roi.x, roi.y, roi.size), (600, 400, 200))
        crop_mask, full_mask = model_mask_to_roi(
            np.ones((64, 64), np.float32),
            roi,
            (800, 600),
        )
        self.assertEqual(crop_mask.size, (200, 200))
        self.assertEqual(full_mask.size, (800, 600))
        self.assertEqual(full_mask.getpixel((0, 0)), 0)
        self.assertEqual(full_mask.getpixel((700, 500)), 255)

    def test_square_roi_is_created_around_mask_with_context(self) -> None:
        mask = np.zeros((600, 800), np.float32)
        mask[200:300, 350:450] = 1
        roi = square_roi_from_mask(mask, padding_ratio=0.25)
        self.assertEqual(roi.size, 150)
        self.assertLessEqual(roi.x, 350)
        self.assertLessEqual(roi.y, 200)
        self.assertGreaterEqual(roi.x + roi.size, 450)
        self.assertGreaterEqual(roi.y + roi.size, 300)


if __name__ == "__main__":
    unittest.main()
