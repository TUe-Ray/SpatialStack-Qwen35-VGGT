import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from qwen_vl.model.controlled_vggt_fusion import (
    CachedVGGTControlledFusion,
    align_vggt_to_qwen_premerger,
)


def _config(candidate):
    return SimpleNamespace(
        controlled_fusion_candidate=candidate,
        controlled_cross_attention_heads=4,
        controlled_fusion_dropout=0.0,
        controlled_projector_hidden_dim=16,
        vision_config=SimpleNamespace(hidden_size=16, spatial_merge_size=2),
        text_config=SimpleNamespace(hidden_size=12),
    )


class ControlledFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid = torch.tensor([[1, 4, 6], [1, 6, 4]], dtype=torch.long, device=self.device)

    def test_alignment_shape_and_native_merger_order(self):
        geometry = torch.randn(2, 37 * 37, 2048, device=self.device)
        aligned = align_vggt_to_qwen_premerger(geometry, self.grid)
        self.assertEqual(tuple(aligned.shape), (48, 2048))

        coordinate = torch.zeros(1, 37 * 37, 2048, device=self.device)
        coordinate[0, :, 0] = torch.arange(37 * 37, device=self.device)
        ordered = align_vggt_to_qwen_premerger(
            coordinate, torch.tensor([[1, 4, 4]], device=self.device)
        )
        row_major = F.interpolate(
            coordinate.reshape(1, 37, 37, 2048).permute(0, 3, 1, 2),
            size=(4, 4),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0).reshape(16, 2048)
        native_order = torch.tensor(
            [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15],
            device=self.device,
        )
        torch.testing.assert_close(ordered, row_major.index_select(0, native_order))

    def test_candidate_a_frame_local_backward_and_reload(self):
        module = CachedVGGTControlledFusion(_config("a_premerger_cross_attn")).to(self.device)
        visual = torch.randn(48, 16, device=self.device, requires_grad=True)
        features = {"23": torch.randn(2, 37 * 37, 2048, device=self.device)}
        output = module.fuse_premerger(visual, features, self.grid)
        self.assertEqual(tuple(output.shape), (48, 16))
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate_a.pt"
            torch.save(module.state_dict(), path)
            restored = CachedVGGTControlledFusion(_config("a_premerger_cross_attn")).to(self.device)
            restored.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            module.eval()
            restored.eval()
            torch.testing.assert_close(
                module.fuse_premerger(visual.detach(), features, self.grid),
                restored.fuse_premerger(visual.detach(), features, self.grid),
            )

    def test_candidate_b_zero_init_two_optimizer_steps_and_reload(self):
        module = CachedVGGTControlledFusion(_config("b_llm_add")).to(self.device)
        features = {
            layer: torch.randn(2, 37 * 37, 2048, device=self.device)
            for layer in ("11", "17", "23")
        }
        initial = module.build_language_residuals(features, self.grid, torch.float32, self.device)
        self.assertEqual(set(initial), {0, 1, 2})
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in initial.values()))
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            residuals = module.build_language_residuals(
                features, self.grid, torch.float32, self.device
            )
            loss = sum((value - 1).square().mean() for value in residuals.values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()))
            optimizer.step()
        updated = module.build_language_residuals(features, self.grid, torch.float32, self.device)
        self.assertTrue(any(torch.count_nonzero(value) > 0 for value in updated.values()))
        restored = CachedVGGTControlledFusion(_config("b_llm_add")).to(self.device)
        restored.load_state_dict(module.state_dict())
        for layer in (0, 1, 2):
            torch.testing.assert_close(
                updated[layer],
                restored.build_language_residuals(features, self.grid, torch.float32, self.device)[layer],
            )


if __name__ == "__main__":
    unittest.main()
