from __future__ import annotations

import copy
from typing import Iterator, Sequence

import torch
import torch.nn as nn
from torch import Tensor


# Single Q-network
class QNetwork(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        last = state_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

        final = self.net[-1]
        nn.init.uniform_(final.weight, -3e-3, 3e-3)
        nn.init.uniform_(final.bias, -3e-3, 3e-3)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """Return Q-values of shape ``(B, 1)``."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# Twin critic: two Q-networks + frozen target copies + Polyak averaging
class TwinCritic(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()

        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)

        # Targets start as exact copies of the online nets.
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Freeze the targets — they're updated only through Polyak averaging.
        # Setting requires_grad=False also prevents an upstream optimizer that
        # accidentally receives target params from doing anything.
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

    # Forward passes
    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q_min(self, state: Tensor, action: Tensor) -> Tensor:

    @torch.no_grad()
    def q_target_min(self, next_state: Tensor, next_action: Tensor) -> Tensor:
        q1_t = self.q1_target(next_state, next_action)
        q2_t = self.q2_target(next_state, next_action)
        return torch.min(q1_t, q2_t)

    # Target-network maintenance
    @torch.no_grad()
    def soft_update(self, tau: float) -> None:
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {tau}.")
        for online, target in (
            (self.q1, self.q1_target),
            (self.q2, self.q2_target),
        ):
            for p_online, p_target in zip(
                online.parameters(), target.parameters()
            ):
                p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)

    @torch.no_grad()
    def hard_update(self) -> None:
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    # Optimizer wiring
    def online_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.q1.parameters()
        yield from self.q2.parameters()



if __name__ == "__main__":
    torch.manual_seed(0)

    state_dim, action_dim, batch = 22, 3, 8
    critic = TwinCritic(state_dim, action_dim)

    s = torch.randn(batch, state_dim)
    a = torch.randn(batch, action_dim)

    q1, q2 = critic(s, a)
    print(f"Q1 shape={tuple(q1.shape)}  mean={q1.mean():+.3f}")
    print(f"Q2 shape={tuple(q2.shape)}  mean={q2.mean():+.3f}")
    print(f"Q1 ≠ Q2? {(not torch.allclose(q1, q2))}   "
          f"(expect True — independent inits)")

    q_min = critic.q_min(s, a)
    assert torch.equal(q_min, torch.min(q1, q2))

   
    qt = critic.q_target_min(s, a)
    print(f"target_min shape={tuple(qt.shape)}  mean={qt.mean():+.3f}")
    assert torch.allclose(qt, torch.min(q1, q2).detach(), atol=1e-6), (
        "Targets should match online nets at init."
    )

    loss = q1.sum() + q2.sum()
    loss.backward()
    assert all(p.grad is None for p in critic.q1_target.parameters())
    assert all(p.grad is None for p in critic.q2_target.parameters())
    print("backward OK — gradients reached online critics, targets untouched.")

    with torch.no_grad():
        for p in critic.online_parameters():
            p.add_(0.5)
    critic.soft_update(tau=0.5)
    sample_online = next(critic.q1.parameters()).data.mean().item()
    sample_target = next(critic.q1_target.parameters()).data.mean().item()
    print(f"after soft_update(0.5): online={sample_online:+.3f}  "
          f"target={sample_target:+.3f}  (target ≈ online - 0.25)")
