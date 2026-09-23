import torch
import torch.nn.functional as F

from qwen_vl.model.supervised_token_loss import supervised_token_causal_loss


def test_supervised_token_loss_matches_full_causal_loss_and_gradients():
    torch.manual_seed(7)
    hidden = torch.randn(2, 9, 12, dtype=torch.float32)
    labels = torch.full((2, 9), -100, dtype=torch.long)
    labels[0, 7:] = torch.tensor([3, 5])
    labels[1, 5:8] = torch.tensor([1, 4, 2])
    weight = torch.randn(19, 12, dtype=torch.float32)

    full_hidden = hidden.clone().requires_grad_()
    full_weight = weight.clone().requires_grad_()
    full_logits = F.linear(full_hidden, full_weight)
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
    full_loss = F.cross_entropy(full_logits.float().reshape(-1, 19), shifted.reshape(-1))
    full_loss.backward()

    sparse_hidden = hidden.clone().requires_grad_()
    sparse_weight = weight.clone().requires_grad_()
    sparse_loss, sparse_logits = supervised_token_causal_loss(
        sparse_hidden, labels, lambda x: F.linear(x, sparse_weight)
    )
    sparse_loss.backward()

    assert sparse_logits.shape == (5, 19)
    torch.testing.assert_close(sparse_loss, full_loss)
    torch.testing.assert_close(sparse_hidden.grad, full_hidden.grad)
    torch.testing.assert_close(sparse_weight.grad, full_weight.grad)
