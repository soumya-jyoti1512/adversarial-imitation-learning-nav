from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import h5py
import numpy as np
import torch
from torch import Tensor


EXPECTED_FORMAT_VERSION = "1.0"


class ExpertBatch(TypedDict):
    state:      Tensor   # (B, state_dim)
    action:     Tensor   # (B, action_dim)
    next_state: Tensor   # (B, state_dim)
    done:       Tensor   # (B, 1)


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class ExpertBuffer:

    def __init__(
        self,
        path: str | Path,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        strict: bool = True,
    ) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Expert dataset not found: {path}")

        with h5py.File(path, "r") as f:
            #Schema validation
            required = {"states", "actions", "next_states", "dones"}
            missing = required - set(f.keys())
            if missing:
                raise ValueError(
                    f"Expert HDF5 missing required datasets: {sorted(missing)}"
                )
            if strict:
                version = _decode_attr(f.attrs.get("format_version", ""))
                if version != EXPECTED_FORMAT_VERSION:
                    raise ValueError(
                        f"format_version mismatch — file reports '{version}', "
                        f"loader expects '{EXPECTED_FORMAT_VERSION}'. "
                        f"Pass strict=False to override."
                    )

            #Eager load
            self._states      = f["states"     ][...].astype(np.float32, copy=False)
            self._actions     = f["actions"    ][...].astype(np.float32, copy=False)
            self._next_states = f["next_states"][...].astype(np.float32, copy=False)
            dones = f["dones"][...].astype(np.float32, copy=False)
            if dones.ndim == 1:
                dones = dones[:, None]
            self._dones = dones

            if "episode_starts" in f:
                self._episode_starts = f["episode_starts"][...].astype(np.int64)
            else:
                self._episode_starts = None

        #Internal consistency
        N, state_dim = self._states.shape
        _, action_dim = self._actions.shape
        if self._next_states.shape != (N, state_dim):
            raise ValueError(
                f"next_states shape {self._next_states.shape} "
                f"does not match states ({N}, {state_dim})"
            )
        if self._dones.shape != (N, 1):
            raise ValueError(
                f"dones shape {self._dones.shape} != ({N}, 1)"
            )
        if N == 0:
            raise ValueError(f"Expert dataset {path} is empty (N = 0).")

        self.path = path
        self.device = torch.device(device)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)

    #Sampling
    def sample(self, batch_size: int) -> ExpertBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        idx = self._rng.integers(0, len(self), size=batch_size)
        return ExpertBatch(
            state=      torch.from_numpy(self._states[idx]     ).to(self.device),
            action=     torch.from_numpy(self._actions[idx]    ).to(self.device),
            next_state= torch.from_numpy(self._next_states[idx]).to(self.device),
            done=       torch.from_numpy(self._dones[idx]      ).to(self.device),
        )

    # Introspection
    def __len__(self) -> int:
        return self._states.shape[0]

    @property
    def num_episodes(self) -> int:
        """Number of trajectories (0 if ``episode_starts`` was not stored)."""
        if self._episode_starts is None:
            return 0
        return int(self._episode_starts.shape[0])

    @property
    def avg_episode_length(self) -> float:
        if self.num_episodes == 0:
            return float("nan")
        return len(self) / self.num_episodes

    # Writer 
    @staticmethod
    def write_hdf5(
        path: str | Path,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        episode_starts: np.ndarray | None = None,
    ) -> None:
        states      = np.asarray(states,      dtype=np.float32)
        actions     = np.asarray(actions,     dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones       = np.asarray(dones,       dtype=np.float32).reshape(-1, 1)

        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError("states and actions must be 2-D arrays")

        N, state_dim = states.shape
        action_dim = actions.shape[1]

        if actions.shape[0] != N:
            raise ValueError(
                f"states ({N} rows) and actions ({actions.shape[0]} rows) "
                f"have mismatched N"
            )
        if next_states.shape != (N, state_dim):
            raise ValueError(
                f"next_states shape {next_states.shape} "
                f"!= ({N}, {state_dim})"
            )
        if dones.shape != (N, 1):
            raise ValueError(f"dones shape {dones.shape} != ({N}, 1)")

        num_episodes = 0
        if episode_starts is not None:
            episode_starts = np.asarray(episode_starts, dtype=np.int64)
            num_episodes = int(episode_starts.shape[0])
            if num_episodes > 0:
                if (episode_starts < 0).any() or (episode_starts >= N).any():
                    raise ValueError(
                        f"episode_starts contains out-of-range index "
                        f"(N={N}, min={episode_starts.min()}, "
                        f"max={episode_starts.max()})"
                    )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("states",      data=states,      compression="gzip")
            f.create_dataset("actions",     data=actions,     compression="gzip")
            f.create_dataset("next_states", data=next_states, compression="gzip")
            f.create_dataset("dones",       data=dones,       compression="gzip")
            if episode_starts is not None:
                f.create_dataset("episode_starts", data=episode_starts)
            f.attrs["state_dim"]      = int(state_dim)
            f.attrs["action_dim"]     = int(action_dim)
            f.attrs["num_episodes"]   = num_episodes
            f.attrs["format_version"] = EXPECTED_FORMAT_VERSION


if __name__ == "__main__":
    import os
    import tempfile

    state_dim, action_dim = 22, 3
    num_episodes = 40

    rng = np.random.default_rng(0)
    ep_lens = rng.integers(150, 250, size=num_episodes)
    N = int(ep_lens.sum())

    states      = rng.standard_normal(size=(N, state_dim)).astype(np.float32)
    actions     = (rng.standard_normal(size=(N, action_dim)) * 0.5).astype(np.float32)
    next_states = rng.standard_normal(size=(N, state_dim)).astype(np.float32)
    dones       = np.zeros((N, 1), dtype=np.float32)

    episode_starts = np.zeros(num_episodes, dtype=np.int64)
    cur = 0
    for i, L in enumerate(ep_lens):
        episode_starts[i] = cur
        cur += int(L)
        dones[cur - 1, 0] = 1.0 

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name

    ExpertBuffer.write_hdf5(
        path, states, actions, next_states, dones, episode_starts
    )

    buf = ExpertBuffer(path, seed=42)
    print(f"loaded:  N={len(buf)}  state_dim={buf.state_dim}  "
          f"action_dim={buf.action_dim}  "
          f"episodes={buf.num_episodes}  "
          f"avg_len={buf.avg_episode_length:.1f}")

    assert len(buf) == N
    assert buf.state_dim == state_dim
    assert buf.action_dim == action_dim
    assert buf.num_episodes == num_episodes
    assert num_episodes > 0
    assert abs(buf.avg_episode_length - N / num_episodes) < 1e-6

    batch = buf.sample(64)
    print(f"batch shapes: "
          f"state={tuple(batch['state'].shape)}  "
          f"action={tuple(batch['action'].shape)}  "
          f"next_state={tuple(batch['next_state'].shape)}  "
          f"done={tuple(batch['done'].shape)}")
    assert batch["state"].shape == (64, state_dim)
    assert batch["action"].shape == (64, action_dim)
    assert batch["next_state"].shape == (64, state_dim)
    assert batch["done"].shape == (64, 1)
    assert torch.isin(batch["done"], torch.tensor([0.0, 1.0])).all()

    true_done_frac = num_episodes / N
    print(f"done fraction: dataset={true_done_frac:.4f}  "
          f"sampled batch={batch['done'].mean():.4f}  "
          f"(should be near each other for large batches)")

    with h5py.File(path, "a") as f:
        del f.attrs["format_version"]
        f.attrs["format_version"] = "0.9"
    try:
        ExpertBuffer(path, strict=True)
    except ValueError as e:
        print(f"strict mode rejected bad version: {str(e)[:80]}…")

    buf_lax = ExpertBuffer(path, strict=False)
    assert len(buf_lax) == N
    print("non-strict load succeeded as expected.")

    with h5py.File(path, "a") as f:
        del f["dones"]
    try:
        ExpertBuffer(path, strict=False)
    except ValueError as e:
        print(f"missing 'dones' rejected: {str(e)[:80]}…")

    os.remove(path)
    print("all checks pass.")
