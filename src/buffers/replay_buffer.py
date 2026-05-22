from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor


class TransitionBatch(TypedDict):
    """Shape contract for a sampled mini-batch."""
    state:      Tensor  # (B, state_dim)
    action:     Tensor  # (B, action_dim)
    next_state: Tensor  # (B, state_dim)
    done:       Tensor  # (B, 1), float32 in {0., 1.}


def _to_np(arr, expected_dim: int, name: str) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"{name} has size {arr.shape[0]}, expected {expected_dim}."
        )
    return arr


class ReplayBuffer:

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")

        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)

        self._states      = np.empty((capacity, state_dim),  dtype=np.float32)
        self._actions     = np.empty((capacity, action_dim), dtype=np.float32)
        self._next_states = np.empty((capacity, state_dim),  dtype=np.float32)
        self._dones       = np.empty((capacity, 1),          dtype=np.float32)

        self._pos = 0    
        self._size = 0    
        self._rng = np.random.default_rng(seed)

    # Insertion
    def add(
        self,
        state,
        action,
        next_state,
        done: bool,
    ) -> None:
        self._states[self._pos]      = _to_np(state,      self.state_dim,  "state")
        self._actions[self._pos]     = _to_np(action,     self.action_dim, "action")
        self._next_states[self._pos] = _to_np(next_state, self.state_dim,  "next_state")
        self._dones[self._pos, 0]    = float(bool(done))

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # Sampling
    def sample(self, batch_size: int) -> TransitionBatch:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        idx = self._rng.integers(0, self._size, size=batch_size)

        return TransitionBatch(
            state=      torch.from_numpy(self._states[idx]     ).to(self.device),
            action=     torch.from_numpy(self._actions[idx]    ).to(self.device),
            next_state= torch.from_numpy(self._next_states[idx]).to(self.device),
            done=       torch.from_numpy(self._dones[idx]      ).to(self.device),
        )

    # Introspection
    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    # Persistence
    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            states=self._states,
            actions=self._actions,
            next_states=self._next_states,
            dones=self._dones,
            pos=np.array(self._pos, dtype=np.int64),
            size=np.array(self._size, dtype=np.int64),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> "ReplayBuffer":
        data = np.load(Path(path))
        capacity, state_dim = data["states"].shape
        _, action_dim = data["actions"].shape

        buf = cls(
            capacity=int(capacity),
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            device=device,
            seed=seed,
        )
        buf._states      = data["states"].copy()
        buf._actions     = data["actions"].copy()
        buf._next_states = data["next_states"].copy()
        buf._dones       = data["dones"].copy()
        buf._pos = int(data["pos"])
        buf._size = int(data["size"])
        return buf



if __name__ == "__main__":
    import os
    import tempfile

    state_dim, action_dim, capacity = 22, 3, 100
    buf = ReplayBuffer(capacity, state_dim, action_dim, seed=42)

    for i in range(150):
        s      = np.full(state_dim,  i,     dtype=np.float32)
        a      = np.random.randn(action_dim).astype(np.float32)
        s_next = np.full(state_dim,  i + 1, dtype=np.float32)
        done   = (i % 25 == 24)
        buf.add(s, a, s_next, done)

    assert len(buf) == capacity, f"size should cap at {capacity}, got {len(buf)}"
    assert buf.is_full
    print(f"buffer:  len={len(buf)}  is_full={buf.is_full}")

    batch = buf.sample(32)
    print(f"batch shapes: "
          f"state={tuple(batch['state'].shape)}  "
          f"action={tuple(batch['action'].shape)}  "
          f"next_state={tuple(batch['next_state'].shape)}  "
          f"done={tuple(batch['done'].shape)}")
    assert batch["state"].shape == (32, state_dim)
    assert batch["action"].shape == (32, action_dim)
    assert batch["next_state"].shape == (32, state_dim)
    assert batch["done"].shape == (32, 1)

    markers = batch["state"][:, 0].numpy()
    print(f"sampled markers: min={markers.min():.0f}  max={markers.max():.0f}  "
          f"(expected in [50, 149])")
    assert markers.min() >= 50, "sampled an overwritten slot"
    assert markers.max() <= 149

    dones = batch["done"].numpy()
    assert np.isin(dones, [0.0, 1.0]).all(), "done must be a 0/1 float"
    print(f"done values are clean (0/1 only), terminal count = {int(dones.sum())}")

    s_markers = batch["state"][:, 0].numpy()
    ns_markers = batch["next_state"][:, 0].numpy()
    assert np.allclose(ns_markers, s_markers + 1), "next_state out of sync"
    print("(s, s') pairing preserved through sampling.")

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        ckpt = f.name
    buf2 = ReplayBuffer(capacity, state_dim, action_dim, seed=42)
    for i in range(150):
        s      = np.full(state_dim,  i,     dtype=np.float32)
        a      = np.random.randn(action_dim).astype(np.float32)
        s_next = np.full(state_dim,  i + 1, dtype=np.float32)
        buf2.add(s, a, s_next, i % 25 == 24)
    buf2.save(ckpt)
    buf3 = ReplayBuffer.load(ckpt, seed=42)
    os.remove(ckpt)

    b2 = buf2.sample(32)
    b3 = buf3.sample(32)
    assert torch.allclose(b2["state"],      b3["state"])
    assert torch.allclose(b2["action"],     b3["action"])
    assert torch.allclose(b2["next_state"], b3["next_state"])
    assert torch.allclose(b2["done"],       b3["done"])
    print("save / load round-trip matches bit-for-bit.")

    empty = ReplayBuffer(10, state_dim, action_dim)
    try:
        empty.sample(1)
    except RuntimeError as e:
        print(f"empty-buffer sample raised as expected: {e}")
