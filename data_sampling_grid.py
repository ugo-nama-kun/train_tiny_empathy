# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
from copy import deepcopy

import gymnasium as gym
import tiny_empathy
from tiny_empathy.wrappers import GridRoomsWrapper

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm


def make_env(env_id, enable_empathy, weight_empathy):
    def thunk():
        env = gym.make(env_id,
                       size=5,
                       enable_empathy=enable_empathy,
                       weight_empathy=weight_empathy)

        env = GridRoomsWrapper(env)
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

        x_in = envs.single_observation_space.shape[0]

        self.network = nn.Sequential(
            layer_init(nn.Linear(x_in, 32)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(32, 32)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(32, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(32, 1), std=1)

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
    experiment = "GridRooms-v0"
    n_samples = 1
    max_time_steps = 2_000

    model_types = ["no", "channel", "reward", "full"]
    model_files = dict()
    for type_ in model_types:
        model_directory = f"data/{experiment}/models/{type_}"
        prefix = f"{type_}-2024"
        model_files[type_] = [model_directory + "/" + f for f in os.listdir(model_directory) if f.startswith(prefix)]

    print(model_files)

    empathy_channels = {
        "no": False,
        "channel": True,
        "reward": False,
        "full": True
    }

    # data file: (n_types, n_models, n_samples, time_steps) matrix
    tmp = {n: np.zeros((n_samples, max_time_steps)).tolist() for n in range(len(model_files[model_types[0]]))}
    tmp2 = {
        "possessor_energy": deepcopy(tmp),
        "action": deepcopy(tmp),
        "have_food": deepcopy(tmp),
        "position": deepcopy(tmp),
        "partner_energy": deepcopy(tmp),
    }
    data = {
        "no": deepcopy(tmp2),
        "channel": deepcopy(tmp2),
        "reward": deepcopy(tmp2),
        "full": deepcopy(tmp2),
    }
    # get data by data["no"]["action"][n_sample][time_step]

    device = torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    weight_empathy = 0.0

    for type_ in model_types:
        for id_model, model_file in enumerate(model_files[type_]):
            # env setup
            envs = gym.vector.SyncVectorEnv(
                [make_env(f"tiny_empathy/{experiment}", empathy_channels[type_], weight_empathy) for i in range(1)],
            )

            agent = Agent(envs, empathy_channels[type_]).to(device)
            agent.load_state_dict(torch.load(model_file))
            agent.eval()

            for n in tqdm(range(n_samples), desc=f"TYPE:{type_}, MODEL: {id_model+1}/{len(model_files[model_types[0]])} "):
                global_step = 0

                while global_step < max_time_steps:
                    seed = np.random.randint(2 ** 32)
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                    next_obs, _ = envs.reset(seed=seed)
                    next_obs = torch.Tensor(next_obs).to(device)
                    next_done = torch.zeros(1).to(device)
                    next_lstm_state = (
                        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
                        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
                    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

                    initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

                    while global_step < max_time_steps:
                        with torch.no_grad():
                            action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)

                        next_obs, reward, next_done, truncations, infos = envs.step(action.cpu().numpy())
                        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                        # print(infos)
                        # print(infos[0][0]["energy"], infos[1][0]["energy"])

                        data[type_]["action"][id_model][n][global_step] = int(action)
                        data[type_]["possessor_energy"][id_model][n][global_step] = float(infos[0][0]["energy"])
                        data[type_]["have_food"][id_model][n][global_step] = int(infos[0][0]["have_food"])
                        data[type_]["position"][id_model][n][global_step] = int(infos[0][0]["position"])
                        data[type_]["partner_energy"][id_model][n][global_step] = float(infos[1][0]["energy"])

                        global_step += 1
                        if any(next_done.tolist()):
                            # print(f"terminal: {global_step}")
                            break

            envs.close()

    import json
    file_path = f"data/{experiment}/data_all.json"
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print("done")
