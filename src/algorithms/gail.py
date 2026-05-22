from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from src.networks.discriminator import Discriminator


class GAILTrainer:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        lr: float = 3e-4,
        r1_coeff: float = 10.0,
        use_tanh: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        if r1_coeff < 0.0:
            raise ValueError(f"r1_coeff must be ≥ 0, got {r1_coeff}")

        self.device = torch.device(device)
        self.r1_coeff = float(r1_coeff)

        self.discriminator = Discriminator(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            use_tanh=use_tanh,
        ).to(self.device)

        self.opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self._update_count = 0

    # Reward query
    def compute_reward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.discriminator.reward(state, action)

    # Update step
    def update(
        self,
        expert_batch: dict[str, Tensor],
        agent_batch: dict[str, Tensor],
    ) -> dict[str, float]:
        # BCE loss + diagnostics
        diag = self.discriminator.loss(
            expert_state=expert_batch["state"],
            expert_action=expert_batch["action"],
            agent_state=agent_batch["state"],
            agent_action=agent_batch["action"],
        )
        bce_loss = diag["loss"]

        # R1 gradient penalty
        r1 = self.discriminator.r1_penalty(
            expert_state=expert_batch["state"],
            expert_action=expert_batch["action"],
            coeff=self.r1_coeff,
        )

        total_loss = bce_loss + r1

        # Optimizer step
        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()

        grad_norm = self._compute_grad_norm()

        self.opt.step()
        self._update_count += 1

        return {
            "bce_loss":    float(bce_loss.detach()),
            "r1_penalty":  float(r1.detach()),
            "total_loss":  float(total_loss.detach()),
            "loss_expert": float(diag["loss_expert"]),
            "loss_agent":  float(diag["loss_agent"]),
            "d_expert":    float(diag["d_expert"]),
            "d_agent":     float(diag["d_agent"]),
            "acc":         float(diag["acc"]),
            "grad_norm":   grad_norm,
        }

    def _compute_grad_norm(self) -> float:
        total_sq = 0.0
        for p in self.discriminator.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().pow(2).sum().item()
        return total_sq ** 0.5

    @property
    def update_count(self) -> int:
        return self._update_count

    # Checkpointing
    def state_dict(self) -> dict:
        return {
            "discriminator": self.discriminator.state_dict(),
            "opt":           self.opt.state_dict(),
            "update_count":  self._update_count,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.discriminator.load_state_dict(sd["discriminator"])
        self.opt.load_state_dict(sd["opt"])
        self._update_count = int(sd.get("update_count", 0))



if __name__ == "__main__":
    import math

    torch.manual_seed(0)

    state_dim, action_dim, B = 22, 3, 64

    trainer = GAILTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        r1_coeff=10.0,
        device="cpu",
    )

    def make_batch(expert: bool) -> dict[str, Tensor]:
        if expert:
            s = torch.randn(B, state_dim) + 1.0
            a = torch.randn(B, action_dim) * 0.3
        else:
            s = torch.randn(B, state_dim) - 1.0
            a = torch.randn(B, action_dim) * 0.8
        return {"state": s, "action": a}

    print(f"{'step':>4}  {'bce':>5}  {'r1':>5}  {'D(exp)':>6}  "
          f"{'D(agn)':>6}  {'acc':>5}  {'|∇|':>5}")
    for step in range(60):
        m = trainer.update(make_batch(expert=True), make_batch(expert=False))
        if step % 15 == 0 or step == 59:
            print(f"{step:>4}  {m['bce_loss']:>5.3f}  "
                  f"{m['r1_penalty']:>5.3f}  "
                  f"{m['d_expert']:>6.3f}  {m['d_agent']:>6.3f}  "
                  f"{m['acc']:>5.3f}  {m['grad_norm']:>5.3f}")
        for k, v in m.items():
            assert math.isfinite(v), f"{k} went non-finite at step {step}"

    assert m["d_expert"] > 0.9
    assert m["d_agent"]  < 0.1
    assert m["acc"]      > 0.95
    assert trainer.update_count == 60
    print(f"\nupdate_count={trainer.update_count}, D separates the two cleanly.")

    eb = make_batch(expert=True)
    ab = make_batch(expert=False)
    r_expert = trainer.compute_reward(eb["state"], eb["action"])
    r_agent  = trainer.compute_reward(ab["state"], ab["action"])
    print(f"r_gail(expert)  mean={r_expert.mean():.3f}  "
          f"r_gail(agent)  mean={r_agent.mean():.3f}  "
          f"(expert ≫ agent expected)")
    assert (r_expert >= 0).all() and (r_agent >= 0).all(), "reward must be ≥ 0"
    assert r_expert.mean() > r_agent.mean(), "reward signal must point at expert"
   
    assert not r_expert.requires_grad

    trainer_no_r1 = GAILTrainer(state_dim, action_dim, r1_coeff=0.0)
    m_no_r1 = trainer_no_r1.update(make_batch(True), make_batch(False))
    assert m_no_r1["r1_penalty"] == 0.0
    print(f"r1_coeff=0 path: r1_penalty={m_no_r1['r1_penalty']:.3f} as expected.")

    sd = trainer.state_dict()
    fresh = GAILTrainer(state_dim, action_dim, r1_coeff=10.0)
    fresh.load_state_dict(sd)
    assert fresh.update_count == trainer.update_count

    sa = torch.cat([eb["state"], eb["action"]], dim=-1)
    with torch.no_grad():
        lp_orig = trainer.discriminator(eb["state"], eb["action"])
        lp_load = fresh.discriminator(eb["state"], eb["action"])
    assert torch.allclose(lp_orig, lp_load, atol=1e-6), "checkpoint mismatched"
    print(f"checkpoint round-trip preserved D bit-for-bit "
          f"(update_count={fresh.update_count}).")

    print("\nall checks pass.")
