# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import tiny_empathy
from tiny_empathy.envs import GridRoomsDecoderLearningEnv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils import SyncVectorDecoderLearningEnv
from wrapper import RecordEpisodeStatisticsDecoderLearning


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
    wandb_group_name: str = "GridRooms-v0"
    """the wandb's group name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "tiny_empathy/GridRooms-v0"
    """the id of the environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.001
    """the learning rate of the optimizer"""
    num_envs: int = 16
    num_test_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 100
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    enable_learning: bool = False
    "if toggled, enable decoder learning"
    decoding_mode: str = "full"
    """ decoding mode; full or affect"""
    weight_empathy: float = 0.5
    """ affective empathy weight """
    dim_emotional_feature: int = 5
    hidden_size_decoder: int = 20

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(args, enc):
    def thunk():
        env = GridRoomsDecoderLearningEnv(
            decoding_mode=args.decoding_mode,
            dim_emotional_feature=args.dim_emotional_feature,
            emotional_encoder=enc,
            weight_empathy=args.weight_empathy
        )
        env = RecordEpisodeStatisticsDecoderLearning(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
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


class EmotionalEncoder(torch.nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, args.hidden_size_decoder),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_size_decoder, args.dim_emotional_feature),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class EmotionalDecoder(torch.nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(args.dim_emotional_feature, args.hidden_size_decoder)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(args.hidden_size_decoder, 1)),
        )

    def forward(self, x):
        return self.model(x)


def test_runs(agent: torch.nn.Module, test_envs: SyncVectorDecoderLearningEnv, device, emotional_decoder):
    agent.eval()

    episode_length = 0.0
    n_runs = len(test_envs.envs)

    not_done_flags = {i: True for i in range(n_runs)}

    next_obs, _ = test_envs.reset(seed=args.seed, emotional_decoder=emotional_decoder)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_test_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_test_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_test_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    while np.any(list(not_done_flags.values())):

        with torch.no_grad():
            action, _, _, _, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)

        next_obs, reward, next_done, truncations, infos = test_envs.step(action.cpu().numpy(), emotional_decoder=emotional_decoder)
        next_done = np.logical_or(next_done, truncations)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        if "final_info" in infos:
            for id_, info in enumerate(infos["final_info"].tolist()):
                if info is not None:
                    if not_done_flags[id_] is True:
                        not_done_flags[id_] = False
                        print(f"TEST: episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                        episode_length += info['episode']['l']

                if np.any(list(not_done_flags.values())) is False:
                    break

    episode_length /= float(n_runs)

    agent.train()

    return episode_length


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    s = ""
    if args.decoding_mode == "full":
        s += "full_inference"
        args.wandb_group_name += "/full_inference"
    elif args.decoding_mode == "affect":
        s += "affect_inference"
        args.wandb_group_name += "/affect_inference"
    else:
        raise ValueError(f"invalid mode: {args.decoding_mode}")

    if args.enable_learning:
        s += "-learn"
        args.wandb_group_name += "-learn"
    else:
        s += "-no_learn"
        args.wandb_group_name += "-no_learn"

    run_name = f"{args.env_id}__{args.exp_name}_{s}_{args.seed}__{int(time.time())}"

    # encoder and decoder settings
    enc = EmotionalEncoder(args)
    enc.eval()

    dec = EmotionalDecoder(args)
    dec.train()

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

    # env setup
    envs = SyncVectorDecoderLearningEnv(
        [make_env(args, enc) for i in range(args.num_envs)],
    )
    test_envs = SyncVectorDecoderLearningEnv(
        [make_env(args, enc) for i in range(args.num_test_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(list(agent.parameters()) + list(dec.parameters()), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed, emotional_decoder=dec)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    for iteration in range(1, args.num_iterations + 1):
        # test
        test_length = test_runs(agent, test_envs, device=device, emotional_decoder=dec)
        writer.add_scalar("test/episodic_length", test_length, global_step)

        # training
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy(), emotional_decoder=dec)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
            #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
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
                loss_ppo = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Interoception self-reconstruction loss
                b_energy = b_obs[:, 0].reshape(-1, 1)
                b_emotional_feature = enc(b_energy).detach()
                b_pred_energy = dec(b_emotional_feature)
                loss_self = nn.functional.mse_loss(b_pred_energy, b_energy)

                loss = loss_ppo + float(args.enable_learning) * loss_self

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
        writer.add_scalar("losses/self_interoception_loss", loss_self, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()

    os.makedirs("models", exist_ok=True)
    p = f"models_inference/{args.env_id.split('/')[1]}"
    os.makedirs(p, exist_ok=True)

    s = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if args.enable_learning:
        s += "-learn"
    else:
        s += "-no_learn"
    if args.decoding_mode == "full":
        PATH = p + f"/full_inference-{s}"
    elif args.decoding_mode == "affect":
        PATH = p + f"/affective_inference-{s}"

    else:
        raise ValueError(f"invalid mode: {args.decoding_mode}")

    os.makedirs(PATH, exist_ok=True)
    torch.save(enc, PATH + "/emotional_encoder.pt")
    torch.save(dec, PATH + "/emotional_decoder.pt")
    torch.save(agent.state_dict(), PATH + "/model.pt")


    writer.close()