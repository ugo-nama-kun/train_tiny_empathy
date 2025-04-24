"""Wrapper that tracks the cumulative rewards and episode lengths."""
import time
from collections import deque
from typing import Optional, Union, Generic, SupportsFloat, Any

import numpy as np

import gymnasium as gym
from gymnasium import Env
from gymnasium.core import WrapperObsType, WrapperActType, ActType
from gymnasium.envs.registration import EnvSpec
from gymnasium.vector.utils import spaces
from pandas.core.missing import clean_interp_method
from pettingzoo.utils.env import AgentID, ObsType, ActionType, ParallelEnv

from pettingzoo.utils.wrappers import BaseParallelWrapper
from tiny_empathy.envs import GridRoomsDecoderLearningEnv, FoodShareDecoderLearningEnv


class RecordParallelEpisodeStatistics(BaseParallelWrapper[AgentID, ObsType, ActionType], gym.utils.RecordConstructorArgs):

    def __init__(self, env: ParallelEnv[AgentID, ObsType, ActionType], deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        BaseParallelWrapper.__init__(self, env)

        self.possible_agents = getattr(env, "possible_agents", 1)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[dict] = None
        self.episode_lengths: Optional[dict] = None
        self.return_queue = {agent_id: deque(maxlen=deque_size) for agent_id in self.possible_agents}
        self.length_queue = {agent_id: deque(maxlen=deque_size) for agent_id in self.possible_agents}
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obss, infos = super().reset(**kwargs)
        self.episode_start_times = {agent_id: np.full(self.num_envs, time.perf_counter(), dtype=np.float32) for agent_id in self.possible_agents}
        self.episode_returns = {agent_id : np.zeros(self.num_envs, dtype=np.float32) for agent_id in self.possible_agents}
        self.episode_lengths = {agent_id: np.zeros(self.num_envs, dtype=np.int32) for agent_id in self.possible_agents}
        return obss, infos

    def step(self, actions):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."

        # multi-agent data structure: [agent_id, num_envs]
        for i, agent_id in enumerate(self.possible_agents):
            self.episode_returns[agent_id] += rewards[agent_id]
            dones = np.logical_or(terminations[agent_id], truncations[agent_id])

            self.episode_lengths[agent_id] += 1

            # calculate done for every agents
            num_dones = np.sum((dones, ))
            if num_dones:
                if "episode" in infos[agent_id] or "_episode" in infos[agent_id]:
                    raise ValueError(
                        "Attempted to add episode stats when they already exist"
                    )
                else:
                    infos[agent_id]["episode"] = {}
                    infos[agent_id]["episode"] = {
                        "r": np.where(dones, self.episode_returns[agent_id], 0.0),
                        "l": np.where(dones, self.episode_lengths[agent_id], 0),
                        "t": np.where(
                            dones,
                            np.round(time.perf_counter() - self.episode_start_times[agent_id], 6),
                            0.0,
                        ),
                    }
                    if self.is_vector_env:
                        infos[agent_id]["_episode"] = {}
                        infos[agent_id]["_episode"] = np.where(dones, True, False)

                self.return_queue[agent_id].extend(self.episode_returns[agent_id][dones])
                self.length_queue[agent_id].extend(self.episode_lengths[agent_id][dones])
                self.episode_count += num_dones
                self.episode_lengths[agent_id][dones] = 0
                self.episode_returns[agent_id][dones] = 0
                self.episode_start_times[agent_id][dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )


class DecoderLearningWrapper(gym.Wrapper):

    def step(self, action, emotional_decoder) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.step(action, emotional_decoder=emotional_decoder)

    def reset(
        self, *, emotional_decoder, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.reset(seed=seed, options=options, emotional_decoder=emotional_decoder)


class RecordEpisodeStatisticsDecoderLearning(DecoderLearningWrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        super().__init__(env)

        self.env = env

        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, emotional_decoder, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(emotional_decoder=emotional_decoder, **kwargs)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32
        )
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    def step(self, action, emotional_decoder, **kwargs):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action, emotional_decoder=emotional_decoder)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
