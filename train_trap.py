"""Basic code which shows what it's like to run PPO on the Pistonball env using the parallel API, this code is inspired by CleanRL.

This code is exceedingly basic, with no logging or weights saving.
The intention was for users to have a (relatively clean) ~200 line file to refer to when they want to design their own learning algorithm.

Author: Jet (https://github.com/jjshoots)
"""

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import tiny_empathy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = np.random.randint(2**32)
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "empathy_hrl"
    """the wandb's project name"""
    wandb_group_name: str = "trap"
    """the wandb's group name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "tiny_empathy/Trap-v0"
    """the id of the environment"""
    total_timesteps: int = 30_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0003
    """the learning rate of the optimizer"""
    num_steps: int = 32 * 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.00
    """coefficient of the entropy"""
    vf_coef: float = 0.3
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    enable_empathy: bool = False
    weight_empathy: float = 0.0

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(enable_empathy, weight_empathy):
    def thunk():
        # env = gym.make("tiny_empathy/Trap-v0",
        #                enable_empathy=enable_empathy,
        #                weight_empathy=weight_empathy,
        #                render_mode="human",
        #                max_episode_steps=5000)

        env = gym.make("tiny_empathy/Trap-v0",
                       enable_empathy=enable_empathy,
                       weight_empathy=weight_empathy,
                       max_episode_steps=5000)

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


def test_runs(agent: torch.nn.Module, test_envs, device):
    agent.eval()

    episode_length = 0.0
    n_runs = 1

    for n in range(n_runs):
        steps = 0.0
        next_obs, _ = test_envs.reset()
        next_obs = batchify_obs(next_obs, device)
        next_done = torch.zeros(2).to(device)
        next_lstm_state = (
            torch.zeros(agent.lstm.num_layers, 2, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, 2, agent.lstm.hidden_size).to(device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

        while True:
            with torch.no_grad():
                action, _, _, _, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)

            next_obs, reward, done, _, info = test_envs.step(unbatchify(action, test_envs))
            next_obs, next_done = batchify_obs(next_obs, device), batchify(done, device)

            steps += 1
            if any(done.values()) is True:
                print(f"TEST: episodic_length={steps}")
                episode_length += steps
                break

    episode_length /= float(n_runs)

    agent.train()
    return episode_length


if __name__ == "__main__":
    """ALGO PARAMS"""
    args = tyro.cli(Args)
    args.batch_size = int(2 * args.num_steps)  # two agents
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    s = f"weight_{args.weight_empathy}_"
    if args.enable_empathy is True and args.weight_empathy > 0:
        s += "full_empathy"
        args.wandb_group_name += "/full_emp"
    elif args.enable_empathy is True and args.weight_empathy == 0:
        s += "emp_channel"
        args.wandb_group_name += "/channel_emp"
    elif args.enable_empathy is False and args.weight_empathy > 0:
        s += "emp_reward"
        args.wandb_group_name += "/reward_emp"
    elif args.enable_empathy is False and args.weight_empathy == 0:
        s += "no_empathy"
        args.wandb_group_name += "/no_emp"
        
    run_name = f"{args.env_id}__{args.exp_name}_{s}_{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    """ ENV SETUP """
    env = make_env(args.enable_empathy, args.weight_empathy)()
    test_env = make_env(args.enable_empathy, args.weight_empathy)()
    num_agents = len(env.possible_agents)
    num_actions = env.action_space.n
    observation_size = env.observation_space.shape

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions, with_empathy_channel=args.enable_empathy).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    obs = torch.zeros((args.num_steps, num_agents) + env.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, num_agents) + env.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, num_agents)).to(device)
    dones = torch.zeros((args.num_steps, num_agents)).to(device)
    values = torch.zeros((args.num_steps, num_agents)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = batchify_obs(next_obs, device)
    next_done = torch.zeros(num_agents).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, num_agents, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, num_agents, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    for iteration in range(1, args.num_iterations + 1):
        # test
        test_length = test_runs(agent, test_env, device=device)
        # writer.add_scalar("test/episodic_error", test_error, global_step)
        writer.add_scalar("test/episodic_length", test_length, global_step)

        # training
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += num_agents
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _, info = env.step(unbatchify(action, env))
            rewards[step] = batchify(reward, device)
            next_obs, next_done = batchify_obs(next_obs, device), batchify(done, device)

            # print(info)
            # print(f"DONES: {done}")
            if any(done.values()) is True:
                # print(f"global_step={global_step}, episodic_length={info['steps']}")
                # # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                # writer.add_scalar("charts/episodic_length", info['steps'], global_step)

                # Reset if done
                next_obs, _ = env.reset(seed=args.seed)
                next_obs = batchify_obs(next_obs, device)
                next_done = torch.zeros(num_agents).to(device)
                next_lstm_state = (
                    torch.zeros(agent.lstm.num_layers, num_agents, agent.lstm.hidden_size).to(device),
                    torch.zeros(agent.lstm.num_layers, num_agents, agent.lstm.hidden_size).to(device),
                )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert num_agents % args.num_minibatches == 0
        envsperbatch = num_agents // args.num_minibatches
        envinds = np.arange(num_agents)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, num_agents)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, num_agents, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    env.close()

    os.makedirs("models", exist_ok=True)
    s = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    p = f"models/{args.env_id.split('/')[1]}"
    os.makedirs(p, exist_ok=True)
    if args.enable_empathy is True and args.weight_empathy > 0:
        os.makedirs(p + f"/full", exist_ok=True)
        PATH = p + f"/full/full-{s}.pt"
    elif args.enable_empathy is True and args.weight_empathy == 0:
        os.makedirs(p + f"/channel", exist_ok=True)
        PATH = p + f"/channel/channel-{s}.pt"
    elif args.enable_empathy is False and args.weight_empathy > 0:
        os.makedirs(p + f"/reward", exist_ok=True)
        PATH = p + f"/reward/reward-{s}.pt"
    elif args.enable_empathy is False and args.weight_empathy == 0:
        os.makedirs(p + f"/no", exist_ok=True)
        PATH = p + f"/no/no-{s}.pt"
    else:
        raise ValueError(f"invalid setting : {(args.enable_empathy, args.weight_empathy)}")

    torch.save(agent.state_dict(), PATH)

    writer.close()