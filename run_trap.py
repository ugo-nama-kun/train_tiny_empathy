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


def make_env(enable_empathy, weight_empathy):
    def thunk():
        env = gym.make("tiny_empathy/Trap-v0",
                       enable_empathy=enable_empathy,
                       weight_empathy=weight_empathy,
                       render_mode="human",
                       max_episode_steps=5000,
                       p_trap=0.0005)

        # env = gym.make("tiny_empathy/Trap-v0",
        #                enable_empathy=enable_empathy,
        #                weight_empathy=weight_empathy,
        #                max_episode_steps=5000)

        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, num_actions, with_empathy_channel):
        super().__init__()

        if with_empathy_channel:
            x_in = 10
        else:
            x_in = 9

        self.network = nn.Sequential(
            layer_init(nn.Linear(x_in, 64)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, 64)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(64, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

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


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, vec)
    obs = obs.transpose(0, -1)
    # convert to torch
    obs = torch.tensor(obs, dtype=torch.float32).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x, dtype=torch.float32).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


if __name__ == "__main__":
    # model_name = "data/Trap-v0/models/reward/reward-2024-06-28-12-40-32.pt"
    # enable_empathy = False

    # model_name = "data/Trap-v0/models/full/full-2024-06-28-12-43-49.pt"
    model_name = "data/Trap-v0/models/channel/channel-2024-06-28-10-48-52.pt"
    enable_empathy = True

    seed = np.random.randint(2 ** 32)
    run_name = f"{'Trap-v0'}__{seed}__{int(time.time())}"
    capture_video = False

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cpu")

    # env setup
    weight_empathy = 0.0
    env = make_env(enable_empathy, weight_empathy)()
    num_agents = len(env.possible_agents)
    num_actions = env.action_space.n
    observation_size = env.observation_space.shape

    agent = Agent(num_actions=num_actions, with_empathy_channel=enable_empathy).to(device)
    agent.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    # agent.eval()

    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=seed)
    next_obs = batchify_obs(next_obs, device)
    next_done = torch.zeros(num_agents).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, num_agents, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, num_agents, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    while True:
        with torch.no_grad():
            action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state,
                                                                                    next_done)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, _, info = env.step(unbatchify(action, env))
        next_obs, next_done = batchify_obs(next_obs, device), batchify(done, device)

        global_step += 1
        if any(next_done.tolist()):
            break

    print("finish: ", global_step)
    print("done")
    env.close()