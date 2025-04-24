from copy import deepcopy
from typing import Dict, Iterable, Callable, Optional, Union, Sequence, List, Tuple, Any

import numpy as np
import torch
import gymnasium as gym

from numpy._typing import NDArray
from sync_vector_ma_env import SyncVectorMAEnv
from gymnasium import Env
from gymnasium.spaces import Space
from gymnasium.vector.utils import concatenate, create_empty_array, iterate, batch_space
from gymnasium.vector.vector_env import VectorEnv


def dict_detach(data_dict: Dict[str, torch.Tensor]):
    return {id_: data_dict[id_].detach() for id_ in data_dict.keys()}


def dict_cpu_numpy(data_dict: Dict[str, torch.Tensor]):
    return {id_: data_dict[id_].cpu().numpy() for id_ in data_dict.keys()}


def dict_tensor(data_dict: Dict[str, torch.Tensor], device: torch.device):
    return {id_: torch.Tensor(data_dict[id_]).to(device) for id_ in data_dict.keys()}


def test_env_single(agent, test_envs, device, render=False):
    agent.eval()

    episode_reward = 0.0
    episode_length = 0.0
    ave_reward = 0.0

    n_runs = len(test_envs.envs)

    not_done_flags = {i: True for i in range(n_runs)}

    obs, info = test_envs.reset()
    obs = torch.Tensor(obs).to(device)
    done = torch.Tensor([False, ] * n_runs).to(device)

    while np.any(list(not_done_flags.values())):

        with torch.no_grad():
            action, _, _ = agent.get_action_and_value(obs, done)

        obs, reward, done, truncated, info = test_envs.step(action.cpu().numpy())
        done = done | truncated

        if render:
            test_envs.envs[0].render()

        obs = torch.Tensor(obs).to(device)
        done = torch.Tensor(done).to(device)

        if np.any(done.cpu().numpy()):
            for i in np.where(info["final_info"])[0]:
                if not_done_flags[i] is True:
                    not_done_flags[i] = False
                    info_ = info["final_info"][i]
                    print(
                        f"TEST: episodic_return={info_['episode']['r']}, episodic_length={info_['episode']['l']}")

                    episode_reward += info_['episode']['r']
                    episode_length += info_['episode']['l']
                    ave_reward += info_['episode']['r'] / info_['episode']['l']

                if np.any(list(not_done_flags.values())) is False:
                    break

    episode_reward /= n_runs
    episode_length /= n_runs
    ave_reward /= n_runs

    agent.train()

    return episode_reward, episode_length, ave_reward


def test_env_multi(agent, test_envs: SyncVectorMAEnv, device, render=False):
    agent.eval()
    possible_agents = test_envs.possible_agents

    episode_reward = {agent_id: 0.0 for agent_id in possible_agents}
    episode_length = {agent_id: 0.0 for agent_id in possible_agents}
    ave_reward = {agent_id: 0.0 for agent_id in possible_agents}

    n_runs = len(test_envs.ma_envs)
    not_done_flags = {i: True for i in range(n_runs)}

    obs, info = test_envs.reset()
    obs = {agent: torch.Tensor(obs[agent]).to(device) for agent in obs.keys()}
    done = {agent: torch.zeros(n_runs).to(device) for agent in possible_agents}

    while np.any(list(not_done_flags.values())):

        with torch.no_grad():
            action, _, _ = agent.get_action_and_value(obs, done)

        obs, reward, done, truncated, infos = test_envs.step(dict_cpu_numpy(action))
        done = done | truncated

        if render:
            test_envs.ma_envs[0].render()

        done = np.logical_or(done, truncated)
        obs, done = dict_tensor(obs, device), dict_tensor(done, device)

        for env_id in range(n_runs):
            if not_done_flags[env_id] is True:
                for agent_id in possible_agents:
                    if "final_info" in infos[agent_id][env_id]:
                        not_done_flags[env_id] = False
                        info = infos[agent_id][env_id]["final_info"]
                        print(
                            f"TEST-{agent_id}: episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                        episode_reward[agent_id] += info['episode']['r']
                        episode_length[agent_id] += info['episode']['l']
                        ave_reward[agent_id] += info['episode']['r'] / info['episode']['l']

    for agent_id in possible_agents:
        episode_reward[agent_id] /= n_runs
        episode_length[agent_id] /= n_runs
        ave_reward[agent_id] /= n_runs

    agent.train()

    return episode_reward, episode_length, ave_reward


"""A synchronous vector environment for decoder learning"""


class VectorDecoderLearningEnv(gym.Env):
    def __init__(
            self,
            num_envs: int,
            observation_space: gym.Space,
            action_space: gym.Space,
    ):
        self.num_envs = num_envs
        self.is_vector_env = True
        self.observation_space = batch_space(observation_space, n=num_envs)
        self.action_space = batch_space(action_space, n=num_envs)

        self.closed = False
        self.viewer = None

        # The observation and action spaces of a single environment are
        # kept in separate properties
        self.single_observation_space = observation_space
        self.single_action_space = action_space

    def reset_async(
            self,
            emotional_decoder=None,
            seed: Optional[Union[int, List[int]]] = None,
            options: Optional[dict] = None,
    ):
        pass

    def reset_wait(
            self,
            emotional_decoder,
            seed: Optional[Union[int, List[int]]] = None,
            options: Optional[dict] = None,
    ):
        raise NotImplementedError("VectorEnv does not implement function")

    def reset(
            self,
            *,
            emotional_decoder=None,
            seed: Optional[Union[int, List[int]]] = None,
            options: Optional[dict] = None,
    ):
        self.reset_async(emotional_decoder=emotional_decoder, seed=seed, options=options)
        return self.reset_wait(emotional_decoder=emotional_decoder, seed=seed, options=options)

    def step_async(self, actions):
        pass

    def step_wait(
            self, emotional_decoder
    ) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        raise NotImplementedError()

    def step(
            self, actions, emotional_decoder=None,
    ) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        self.step_async(actions)
        return self.step_wait(emotional_decoder=emotional_decoder)

    def call_async(self, name, *args, **kwargs):
        pass

    def call_wait(self, **kwargs) -> List[Any]:  # type: ignore
        pass

    def call(self, name: str, *args, **kwargs) -> List[Any]:
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def get_attr(self, name: str):
        return self.call(name)

    def set_attr(self, name: str, values: Union[list, tuple, object]):
        pass

    def close_extras(self, **kwargs):
        pass

    def close(self, **kwargs):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras(**kwargs)
        self.closed = True

    def _add_info(self, infos: dict, info: dict, env_num: int) -> dict:
        for k in info.keys():
            if k not in infos:
                info_array, array_mask = self._init_info_arrays(type(info[k]))
            else:
                info_array, array_mask = infos[k], infos[f"_{k}"]

            info_array[env_num], array_mask[env_num] = info[k], True
            infos[k], infos[f"_{k}"] = info_array, array_mask
        return infos

    def _init_info_arrays(self, dtype: type) -> Tuple[np.ndarray, np.ndarray]:
        if dtype in [int, float, bool] or issubclass(dtype, np.number):
            array = np.zeros(self.num_envs, dtype=dtype)
        else:
            array = np.zeros(self.num_envs, dtype=object)
            array[:] = None
        array_mask = np.zeros(self.num_envs, dtype=bool)
        return array, array_mask

    def __del__(self):
        if not getattr(self, "closed", True):
            self.close()

    def __repr__(self) -> str:
        if self.spec is None:
            return f"{self.__class__.__name__}({self.num_envs})"
        else:
            return f"{self.__class__.__name__}({self.spec.id}, {self.num_envs})"


class SyncVectorDecoderLearningEnv(VectorDecoderLearningEnv):
    def __init__(
            self,
            env_fns: Iterable[Callable[[], Env]],
            observation_space: Space = None,
            action_space: Space = None,
            copy: bool = True,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: iterable of callable functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
        """
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            num_envs=len(self.envs),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        """Sets the seed in all sub-environments.

        Args:
            seed: The seed
        """
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

    def reset_wait(
            self,
            emotional_decoder: torch.nn.Module,
            seed: Optional[Union[int, List[int]]] = None,
            options: Optional[dict] = None,
    ):
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            seed: The reset environment seed
            options: Option information for the environment reset

        Returns:
            The reset observation of the environment and reset information
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._terminateds[:] = False
        self._truncateds[:] = False
        observations = []
        infos = {}
        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options

            observation, info = env.reset(emotional_decoder=emotional_decoder, **kwargs)
            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step_async(self, actions):
        """Sets :attr:`_actions` for use by the :meth:`step_wait` by converting the ``actions`` to an iterable version."""
        self._actions = iterate(self.action_space, actions)

    def step_wait(self, emotional_decoder) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action, emotional_decoder)

            if self._terminateds[i] or self._truncateds[i]:
                old_observation, old_info = observation, info
                observation, info = env.reset(emotional_decoder)
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )

    def call(self, name, *args, **kwargs) -> tuple:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name: str, values: Union[list, tuple, Any]):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self) -> bool:
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        return True
