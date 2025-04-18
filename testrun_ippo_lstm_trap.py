
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro

from tiny_empathy.envs import TrapEnvPZ

from ippo_lstm import IPPO_LSTM
from sync_vector_ma_env import SyncVectorMAEnv
from utils import dict_detach, dict_cpu_numpy, dict_tensor
from wrapper import RecordParallelEpisodeStatistics


@dataclass
class Args:
    # Algorithm specific arguments
    capture_video: bool = False
    # remove agent name and .pth to identify model path
    # file_path = "models_trap_pz/TrapPZ__train_ippo_lstm_trap__no__5__1744889249/final"
    file_path: str = "models_trap_pz/TrapPZ__train_ippo_lstm_trap__affective__1__1744889249/final"
    # file_path = "models_trap_pz/TrapPZ__train_ippo_lstm_trap__cognitive__5__1744889249/final"
    # file_path = "models_trap_pz/TrapPZ__train_ippo_lstm_trap__full__5__1744889249/final"
    seed: int = 42
    torch_deterministic: bool = True
    num_envs = 1


def make_env(cognitive_empathy, weight_affective_empathy):
    def thunk():
        env = TrapEnvPZ(render_mode="human",
                        cognitive_empathy=cognitive_empathy,
                        weight_affective_empathy=weight_affective_empathy,
                        # p_trap=0.005,
        )
        env = RecordParallelEpisodeStatistics(env)
        return env

    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if "no" in args.file_path:
        # env setup
        print("MODE: no")
        ma_envs = SyncVectorMAEnv(
            [make_env(cognitive_empathy=False, weight_affective_empathy=0.0)],
        )
    elif "affective" in args.file_path:
        print("MODE: affective")
        ma_envs = SyncVectorMAEnv(
            [make_env(cognitive_empathy=False, weight_affective_empathy=0.5)],
        )
    elif "cognitive" in args.file_path:
        print("MODE: cognitive")
        ma_envs = SyncVectorMAEnv(
            [make_env(cognitive_empathy=True, weight_affective_empathy=0.0)],
        )
    elif "full" in args.file_path:
        print("MODE: full")
        ma_envs = SyncVectorMAEnv(
            [make_env(cognitive_empathy=True, weight_affective_empathy=0.5)],
        )

    for action_space in ma_envs.single_joint_action_space.values():
        assert isinstance(action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # agent = Agent(envs).to(device)
    ippo_agent = IPPO_LSTM(ma_envs, device, args, run_name="test", test=True)
    ippo_agent.load_model(args.file_path)

    while True:
        next_obs, _ = ma_envs.reset(seed=args.seed)
        next_obs = {agent: torch.Tensor(next_obs[agent]).to(device) for agent in next_obs.keys()}
        next_done = {agent: torch.zeros(args.num_envs).to(device) for agent in ma_envs.ma_envs[0].possible_agents}
        ippo_agent.reset_lstm_state()

        done = False
        step = 0
        while not done:
            # ALGO LOGIC: action logic
            action, logprob, values = ippo_agent.get_action_and_value(next_obs, next_done)

            # copy previous obs and done
            next_done_prev, next_obs_prev = dict_detach(next_done), dict_detach(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = ma_envs.step(dict_cpu_numpy(action))
            ma_envs.ma_envs[0].render()

            next_done = np.logical_or(terminations, truncations)
            next_obs, next_done = dict_tensor(next_obs, device), dict_tensor(next_done, device)
            done = terminations[ma_envs.possible_agents[0]]
            step += 1
        print(f"steps: {step}")
