from __future__ import annotations

import torch
from torch import Tensor


class HybridReward:
    def __init__(
        self,
        lambda_goal: float = 0.3,
        lambda_collision: float = 0.5,
        collision_penalty: float = 1.0,
        collision_threshold: float = 0.2,
        lidar_slice: slice = slice(0, 20),
        goal_slice: slice = slice(20, 22),
    ) -> None:
        if collision_penalty < 0.0:
            raise ValueError(
                f"collision_penalty is a magnitude (positive); the "
                f"negative sign is applied internally. Got {collision_penalty}."
            )
        if collision_threshold <= 0.0:
            raise ValueError(
                f"collision_threshold must be > 0, got {collision_threshold}."
            )

        self.lambda_goal = float(lambda_goal)
        self.lambda_collision = float(lambda_collision)
        self.collision_penalty = float(collision_penalty)
        self.collision_threshold = float(collision_threshold)
        self.lidar_slice = lidar_slice
        self.goal_slice = goal_slice

    # Individual components
    def r_goal(self, state: Tensor, next_state: Tensor) -> Tensor:
        d_curr = torch.linalg.vector_norm(
            state[..., self.goal_slice], dim=-1, keepdim=True
        )
        d_next = torch.linalg.vector_norm(
            next_state[..., self.goal_slice], dim=-1, keepdim=True
        )
        return d_curr - d_next

    def r_collision(self, next_state: Tensor) -> Tensor:
        lidar = next_state[..., self.lidar_slice]
        min_dist = lidar.min(dim=-1, keepdim=True).values
        unsafe = (min_dist < self.collision_threshold).to(next_state.dtype)
        return -self.collision_penalty * unsafe

    # Full combined reward
    def compute(
        self,
        state: Tensor,
        next_state: Tensor,
        r_gail: Tensor,
    ) -> dict[str, Tensor]:
        r_goal_t = self.r_goal(state, next_state)
        r_coll_t = self.r_collision(next_state)
        r_total = (
            r_gail
            + self.lambda_goal * r_goal_t
            + self.lambda_collision * r_coll_t
        )
        return {
            "r_gail": r_gail,
            "r_goal": r_goal_t,
            "r_collision": r_coll_t,
            "r_total": r_total,
        }

    def __repr__(self) -> str:
        return (
            f"HybridReward(λ_goal={self.lambda_goal}, "
            f"λ_collision={self.lambda_collision}, "
            f"C={self.collision_penalty}, "
            f"ε={self.collision_threshold} m)"
        )


if __name__ == "__main__":
    state_dim = 22

    hr = HybridReward()
    print(repr(hr))
    assert hr.lambda_goal == 0.3
    assert hr.lambda_collision == 0.5
    assert hr.collision_penalty == 1.0
    assert hr.collision_threshold == 0.2

    state = torch.zeros(1, state_dim)
    next_state = torch.zeros(1, state_dim)
    state[0, :20] = 1.0        
    next_state[0, :20] = 1.0
    state[0, 20:22] = torch.tensor([3.0, 4.0])
    next_state[0, 20:22] = torch.tensor([0.0, 0.0])
    rg = hr.r_goal(state, next_state)
    print(f"r_goal: moved (3, 4) → (0, 0): {rg.item():+.3f}  (expected +5.000)")
    assert abs(rg.item() - 5.0) < 1e-6

    state[0, 20:22] = torch.tensor([1.0, 0.0])
    next_state[0, 20:22] = torch.tensor([2.0, 0.0])
    rg = hr.r_goal(state, next_state)
    print(f"r_goal: moved (1, 0) → (2, 0): {rg.item():+.3f}  (expected -1.000)")
    assert abs(rg.item() + 1.0) < 1e-6

    safe = torch.ones(1, state_dim)      
    rc = hr.r_collision(safe)
    print(f"r_collision: all LiDAR ≥ 1.0 m: {rc.item():+.3f}  (expected  0.000)")
    assert rc.item() == 0.0

    unsafe = torch.ones(1, state_dim)
    unsafe[0, 5] = 0.10                   
    rc = hr.r_collision(unsafe)
    print(f"r_collision: one beam 0.10 m  : {rc.item():+.3f}  (expected -1.000)")
    assert rc.item() == -1.0

    edge = torch.ones(1, state_dim)
    edge[0, 5] = 0.20
    rc = hr.r_collision(edge)
    print(f"r_collision: beam exactly ε  : {rc.item():+.3f}  (expected  0.000)")
    assert rc.item() == 0.0

    batch = 8
    rng = torch.Generator().manual_seed(0)
    s = 0.5 + torch.rand(batch, state_dim, generator=rng)
    s_next = 0.5 + torch.rand(batch, state_dim, generator=rng)
    r_gail = torch.rand(batch, 1, generator=rng) * 5.0

    out = hr.compute(s, s_next, r_gail)
    print(f"compute returns keys: {sorted(out.keys())}")
    assert set(out) == {"r_gail", "r_goal", "r_collision", "r_total"}
    for name, tensor in out.items():
        assert tensor.shape == (batch, 1), (
            f"{name} has shape {tuple(tensor.shape)}, expected ({batch}, 1)"
        )

    expected = (
        out["r_gail"]
        + hr.lambda_goal * out["r_goal"]
        + hr.lambda_collision * out["r_collision"]
    )
    assert torch.allclose(out["r_total"], expected, atol=1e-7)
    print("r_total matches λ₁·r_goal + λ₂·r_collision + r_gail.")

    mixed = torch.ones(4, state_dim)
    mixed[1, 7] = 0.05   
    mixed[3, 0] = 0.15  
    rc = hr.r_collision(mixed)
    print(f"r_collision batch flags: {rc.squeeze(-1).tolist()}  "
          f"(expected [0, -1, 0, -1])")
    assert rc.squeeze(-1).tolist() == [0.0, -1.0, 0.0, -1.0]

    print("all checks pass.")
