from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Discriminator(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        use_tanh: bool = True,
    ) -> None:
        super().__init__()
        activation: type[nn.Module] = nn.Tanh if use_tanh else nn.ReLU

        layers: list[nn.Module] = []
        last = state_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

        final = self.net[-1]
        nn.init.uniform_(final.weight, -3e-3, 3e-3)
        nn.init.uniform_(final.bias, -3e-3, 3e-3)

    # Forward passes
    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def prob(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(state, action))

    @torch.no_grad()
    def reward(self, state: Tensor, action: Tensor) -> Tensor:
        logits = self.forward(state, action)
        return F.softplus(logits)

    # Training losses
    def loss(
        self,
        expert_state: Tensor,
        expert_action: Tensor,
        agent_state: Tensor,
        agent_action: Tensor,
    ) -> dict[str, Tensor]:
        
        expert_logits = self.forward(expert_state, expert_action)
        agent_logits = self.forward(agent_state, agent_action)

        loss_expert = F.softplus(-expert_logits).mean()
        loss_agent = F.softplus(agent_logits).mean()
        loss = loss_expert + loss_agent

        with torch.no_grad():
            d_expert = torch.sigmoid(expert_logits).mean()
            d_agent = torch.sigmoid(agent_logits).mean()
            acc_expert = (expert_logits > 0).float().mean()
            acc_agent = (agent_logits < 0).float().mean()
            acc = 0.5 * (acc_expert + acc_agent)

        return {
            "loss": loss,
            "loss_expert": loss_expert.detach(),
            "loss_agent": loss_agent.detach(),
            "d_expert": d_expert,
            "d_agent": d_agent,
            "acc": acc,
        }

    def r1_penalty(
        self,
        expert_state: Tensor,
        expert_action: Tensor,
        coeff: float = 10.0,
    ) -> Tensor:
        
        if coeff <= 0.0:
            return torch.zeros((), device=expert_state.device)

        s = expert_state.detach().requires_grad_(True)
        a = expert_action.detach().requires_grad_(True)
        logits = self.forward(s, a)

        grads = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=(s, a),
            create_graph=True,
        )
        grad_sq = sum((g.flatten(1) ** 2).sum(dim=1) for g in grads)
        return 0.5 * coeff * grad_sq.mean()



if __name__ == "__main__":
    torch.manual_seed(0)

    state_dim, action_dim, batch = 22, 3, 64
    disc = Discriminator(state_dim, action_dim)
    opt = torch.optim.Adam(disc.parameters(), lr=1e-3)

    expert_s = torch.randn(batch, state_dim) + 1.0
    expert_a = torch.randn(batch, action_dim) * 0.3
    agent_s = torch.randn(batch, state_dim) - 1.0
    agent_a = torch.randn(batch, action_dim) * 0.8

    print(f"{'step':>4}  {'loss':>6}  {'D(exp)':>6}  {'D(agn)':>6}  {'acc':>5}")
    for step in range(60):
        diag = disc.loss(expert_s, expert_a, agent_s, agent_a)
        total = diag["loss"] + disc.r1_penalty(expert_s, expert_a, coeff=10.0)
        opt.zero_grad()
        total.backward()
        opt.step()
        if step % 15 == 0 or step == 59:
            print(f"{step:>4}  {diag['loss'].item():>6.3f}  "
                  f"{diag['d_expert'].item():>6.3f}  "
                  f"{diag['d_agent'].item():>6.3f}  "
                  f"{diag['acc'].item():>5.3f}")

    assert diag["d_expert"].item() > 0.9, "expected D(expert) > 0.9 after training"
    assert diag["d_agent"].item() < 0.1, "expected D(agent)  < 0.1 after training"
    assert diag["acc"].item() > 0.95, "expected accuracy > 0.95 after training"
    print("\nDiscriminator separates the two distributions cleanly.")

    r = disc.reward(agent_s, agent_a)
    print(f"r_gail(agent): shape={tuple(r.shape)}  "
          f"min={r.min():.3f}  mean={r.mean():.3f}  max={r.max():.3f}")
    assert (r >= 0).all(), "softplus(x) must be non-negative"

    r_expert = disc.reward(expert_s, expert_a)
    print(f"r_gail(expert): mean={r_expert.mean():.3f}  "
          f"(should be >> r_gail(agent))")
    assert r_expert.mean() > r.mean(), "expert reward must exceed agent reward"
    print("Reward signal points in the correct direction.")
