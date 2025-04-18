# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import tiny_empathy
from tiny_empathy.wrappers import FoodShareWrapper

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical


def make_env(env_id, idx, capture_video, run_name, enable_empathy, weight_empathy):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id,
                           enable_empathy=enable_empathy,
                           weight_empathy=weight_empathy,
                           render_mode="human")

        env = FoodShareWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, with_empathy_channel):
        super().__init__()

        if with_empathy_channel:
            x_in = 2
        else:
            x_in = 1

        self.network = nn.Sequential(
            layer_init(nn.Linear(x_in, 16)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(16, 16)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(16, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(16, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


if __name__ == "__main__":
    model_name = "models/2024-05-18-01-40-21/agent_full_empathy.pt"
    enable_empathy = True

    # model_name = "models/2024-05-18-01-41-16/agent_emp_reward.pt"
    # enable_empathy = False

    # model_name = "models/2024-05-18-01-40-52/agent_emp_channel.pt"
    # enable_empathy = True

    # model_name = "models/2024-05-18-01-41-40/agent_no_empathy.pt"
    # enable_empathy = False

    seed = 10
    run_name = f"{'FoodShare-v0'}__{seed}__{int(time.time())}"
    capture_video = False

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cpu")

    # env setup
    weight_empathy = 0.0
    envs = gym.vector.SyncVectorEnv(
        [make_env("tiny_empathy/FoodShare-v0", i, capture_video, run_name, enable_empathy, weight_empathy) for i in range(1)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, enable_empathy).to(device)
    agent.load_state_dict(torch.load(model_name))
    agent.eval()

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

    for t in range(1000):
        with torch.no_grad():
            action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)

        next_obs, reward, next_done, truncations, infos = envs.step(action.cpu().numpy())
        envs.envs[0].render()
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        print(infos[0], infos[1], action)

        if any(next_done.tolist()):
            print(f"terminal: {t}")
            break

    print("finish: ", t)
    print("done")
    envs.close()