"""VERA20k direct-command environments."""
import gymnasium as gym

from .env import VeraEnv, VeraBatch

gym.register("VERA20k-v0", entry_point="vera20k_agent.env:VeraEnv")

__all__ = ["VeraEnv", "VeraBatch"]
