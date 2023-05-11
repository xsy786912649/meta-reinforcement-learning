
import os
os.add_dll_directory("C://Users//lenovo//.mujoco//mjpro150//bin")
import sys  # 导入sys模块
sys.setrecursionlimit(10000) 

import random
import math
import time

from copy import deepcopy

import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c, trpo
from cherry.models.robotics import LinearValue
from tqdm import tqdm

import learn2learn as l2l

import torch as th
import torch.nn as nn
from torch import autograd
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter

env = gym.make('HalfCheetahForwardBackward-v1')
env.reset()

EPSILON = 1e-6

def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


class DiagNormalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None, activation='relu', device='cpu'):
        super(DiagNormalPolicy, self).__init__()
        self.device = device
        if hiddens is None:
            hiddens = [100, 100]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.mean = nn.Sequential(*layers)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))

    # def forward(self, state):
    #     state = state.to(self.device, non_blocking=True)
    #     loc = self.mean(state)
    #     scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
    #     return Normal(loc=loc, scale=scale)

    def density(self, state):
        state = state.to(self.device, non_blocking=True)
        loc = self.mean(state)
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    def forward(self, state):
        density = self.density(state)
        action = density.sample()
        return action

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, env, hidden_size=100):
        super().__init__()
        self.input_size = env.observation_space.shape[0]
        self.actor_output_size = env.action_space.shape[0]

        self.l1 = layer_init(nn.Linear(self.input_size, hidden_size))
        self.l2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.output = layer_init(nn.Linear(hidden_size, self.actor_output_size), std=0.01)
        self.activation = nn.ReLU()
        self.distribution = ch.distributions.ActionDistribution(env)

    def forward(self, x):
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.output(x)
        mass = self.distribution(x)

        return mass
    
class Critic(nn.Module):
    def __init__(self, env, lr, hidden_size=32):
        super().__init__()
        self.input_size = env.observation_space.shape[0]
        self.critic_output_size = 1

        self.l1 = layer_init(nn.Linear(self.input_size, hidden_size))
        self.l2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.critic_head = layer_init(nn.Linear(hidden_size, self.critic_output_size), std=1.)
        self.activation = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    def forward(self, x):
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        value = self.critic_head(x)

        return value
    
class MAMLTRPO():
    def __init__(self, env_name,
                 actor_class=Actor, critic_class=Critic, 
                 actor_args=dict(), critic_args=dict(),
                 adapt_lr=0.1, meta_lr=1.0, 
                 adapt_steps=3,
                 adapt_batch_size=20, meta_batch_size=20,
                 gamma=0.95, tau=1.0,
                 backtrack_factor=0.5, ls_max_steps=15, max_kl=0.01,
                 num_workers=5,
                 seed=42,
                 device=None, name="MAMLTRPO", tensorboard_log="./logs"):
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(seed)
        else:
            self.device = torch.device("cpu")
        if device:
            self.device = torch.device(device)
        print("Running on: " + str(self.device))

        def make_env():
            env = gym.make(env_name)
            env = ch.envs.ActionSpaceScaler(env)
            return env

        env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
        env.seed(seed)
        env.set_task(env.sample_tasks(1)[0])
        self.env = ch.envs.Torch(env)

        self.gamma = gamma
        self.tau = tau
        self.adapt_lr = adapt_lr
        self.meta_lr = meta_lr
        self.adapt_steps = adapt_steps
        self.adapt_batch_size = adapt_batch_size
        self.meta_batch_size = meta_batch_size
        self.backtrack_factor = backtrack_factor
        self.ls_max_steps = ls_max_steps
        self.max_kl = max_kl
        self.global_iteration = 0

        # self.policy = Actor(env, **actor_args).to(device)
        # self.baseline = Critic(env, lr=0.001, **critic_args).to(device)
        self.policy = DiagNormalPolicy(self.env.state_size, self.env.action_size, device=self.device)
        self.baseline = LinearValue(self.env.state_size, self.env.action_size)

        self.policy.to(self.device)
        self.baseline.to(self.device)

        if tensorboard_log is not None:
            self.run_name = name + "_" + str(int(time.time()))
            self.writer = SummaryWriter(f"{tensorboard_log}/{self.run_name}")
        else:
            self.writer = None

    
    def save(self, path="./"):
        torch.save(self.baseline.state_dict(), path + "/baseline.pt")
        torch.save(self.policy.state_dict(), path + "/policy.pt")


    def load(self, path="./"):
        self.baseline.load_state_dict(torch.load(path + "/baseline.pt"))
        self.policy.load_state_dict(torch.load(path + "/policy.pt"))


    def collect_steps(self, policy, n_episodes):
        self.env.reset()
        task = ch.envs.Runner(self.env)
        replay = task.run(policy, episodes=n_episodes).to(self.device)

        returns = ch.td.discount(self.gamma, replay.reward(), replay.done())
        self.baseline.fit(replay.state(), returns)
        values = self.baseline(replay.state())
        next_values = self.baseline(replay.next_state())
        bootstraps = values * (1.0 - replay.done()) + next_values * replay.done()
        next_value = torch.zeros(1, device=values.device)
        advantages = ch.pg.generalized_advantage(tau=self.tau,
                                        gamma=self.gamma,
                                        rewards=replay.reward(),
                                        dones=replay.done(),
                                        values=bootstraps,
                                        next_value=next_value)
        advantages = ch.normalize(advantages, epsilon=1e-8).detach()

        for i, sars in enumerate(replay):
            sars.returns = returns[i]
            sars.advantage = advantages[i]

        return replay


    # def collect_steps(self, policy, n_episodes):
    #     # replay = ch.ExperienceReplay(device=self.device)
    #     # for i in range(n_episodes):
    #     #     state = self.env.reset()

    #     #     while True:
    #     #         with torch.no_grad():
    #     #             mass = policy(state)
    #     #         action = mass.sample()
    #     #         log_prob = mass.log_prob(action).mean(dim=1, keepdim=True)
    #     #         next_state, reward, done, _ = self.env.step(action)

    #     #         replay.append(state,
    #     #                     action,
    #     #                     reward,
    #     #                     next_state,
    #     #                     done,
    #     #                     log_prob=log_prob)
                
    #     #         if done.any():
    #     #             break

    #     #         state = next_state
        
    #     self.env.reset()
    #     task = ch.envs.Runner(self.env)
    #     replay = task.run(policy, episodes=n_episodes).to(self.device)

    #     with torch.no_grad():
    #         next_state_value = self.baseline(replay[-1].next_state)
    #     values = self.baseline(replay.state())

    #     advantages = ch.generalized_advantage(self.gamma,
    #                                             self.tau,
    #                                             replay.reward(),
    #                                             replay.done(),
    #                                             values.detach(),
    #                                             next_state_value)
    #     returns = advantages + values.detach()
    #     advantages = ch.normalize(advantages, epsilon=1e-8)

    #     for i, sars in enumerate(replay):
    #         sars.returns = returns[i]
    #         sars.advantage = advantages[i]

    #     # value_loss = a2c.state_value_loss(returns, values)
    #     # self.baseline.optimizer.zero_grad()
    #     # value_loss.backward()
    #     # self.baseline.optimizer.step()

    #     self.baseline.fit(replay.state(), returns)
    #     return replay


    def maml_a2c_loss(self, train_episodes, learner):
        # Update policy and baseline
        states = train_episodes.state()
        actions = train_episodes.action()
        density = learner.density(states)
        log_probs = density.log_prob(actions).mean(dim=1, keepdim=True)

        advantages = train_episodes.advantage()
        return a2c.policy_loss(log_probs, train_episodes.advantage())


    def fast_adapt(self, clone, train_episodes, first_order=False):
        second_order = not first_order
        loss = self.maml_a2c_loss(train_episodes, clone)
        gradients = autograd.grad(loss,
                                clone.parameters(),
                                retain_graph=second_order,
                                create_graph=second_order)
        return l2l.algorithms.maml.maml_update(clone, self.adapt_lr, gradients)


    def meta_surrogate_loss(self, iteration_replays, iteration_policies, policy):
        mean_loss = 0.0
        mean_kl = 0.0
        for task_replays, old_policy in tqdm(zip(iteration_replays, iteration_policies),
                                            total=len(iteration_replays),
                                            desc='Surrogate Loss',
                                            leave=False):
            train_replays = task_replays[:-1]
            valid_episodes = task_replays[-1]
            new_policy = l2l.clone_module(policy)

            # Fast Adapt
            for train_episodes in train_replays:
                new_policy = self.fast_adapt(new_policy, train_episodes, first_order=False)

            # Useful values
            states = valid_episodes.state()
            actions = valid_episodes.action()

            # Compute KL
            old_densities = old_policy.density(states)
            new_densities = new_policy.density(states)
            kl = kl_divergence(new_densities, old_densities).mean()
            mean_kl += kl

            # Compute Surrogate Loss
            advantages = valid_episodes.advantage()
            old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
            new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
            mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)
        mean_kl /= len(iteration_replays)
        mean_loss /= len(iteration_replays)
        return mean_loss, mean_kl


    def meta_optimize(self, iteration_replays, iteration_policies):
        # Compute CG step direction
        old_loss, old_kl = self.meta_surrogate_loss(iteration_replays, iteration_policies, self.policy)

        grad = autograd.grad(old_loss,
                                self.policy.parameters(),
                                retain_graph=True)
        grad = parameters_to_vector([g.detach() for g in grad])
        Fvp = trpo.hessian_vector_product(old_kl, self.policy.parameters())
        step = trpo.conjugate_gradient(Fvp, grad)
        shs = 0.5 * torch.dot(step, Fvp(step))
        lagrange_multiplier = torch.sqrt(shs / self.max_kl)
        step = step / lagrange_multiplier
        step_ = [torch.zeros_like(p.data) for p in self.policy.parameters()]
        vector_to_parameters(step, step_)
        step = step_
        del old_kl, Fvp, grad
        old_loss.detach_()

        # Line-search
        for ls_step in range(self.ls_max_steps):
            stepsize = self.backtrack_factor ** ls_step * self.meta_lr
            clone = deepcopy(self.policy)
            for p, u in zip(clone.parameters(), step):
                p.data.add_(-stepsize, u.data)
            new_loss, kl = self.meta_surrogate_loss(iteration_replays, iteration_policies, clone)
            if new_loss < old_loss and kl < self.max_kl:
                for p, u in zip(self.policy.parameters(), step):
                    p.data.add_(-stepsize, u.data)
                break

        if self.writer is not None:
            self.writer.add_scalar("loss", new_loss, self.global_iteration)
            self.writer.add_scalar("kl", kl, self.global_iteration)


    def train(self, num_iterations=100):
        for iteration in range(num_iterations):
            self.global_iteration += 1
            iteration_reward = 0.0
            iteration_replays = []
            iteration_policies = []

            for task_config in tqdm(self.env.sample_tasks(self.meta_batch_size), leave=False, desc='Data'):
                clone = deepcopy(self.policy)
                self.env.set_task(task_config)
                task_replay = []

                # Fast Adapt
                for step in range(self.adapt_steps):
                    train_episodes = self.collect_steps(clone, n_episodes=self.adapt_batch_size)
                    self.fast_adapt(clone, train_episodes, first_order=True)
                    # self.fast_adapt(clone, train_episodes, first_order=True)
                    task_replay.append(train_episodes)

                # Compute Validation Loss
                valid_episodes = self.collect_steps(clone, n_episodes=self.adapt_batch_size)
                task_replay.append(valid_episodes)
                iteration_reward += valid_episodes.reward().sum().item() / self.adapt_batch_size
                iteration_replays.append(task_replay)
                iteration_policies.append(clone)

            # Print statistics
            print('Iteration', self.global_iteration)
            adaptation_reward = iteration_reward / self.meta_batch_size
            print('adaptation_reward', adaptation_reward)

            if self.writer is not None:
                self.writer.add_scalar("adaptation_reward", adaptation_reward, self.global_iteration)

            self.meta_optimize(iteration_replays, iteration_policies)


if __name__ == '__main__':
    metalearner = MAMLTRPO("HalfCheetahForwardBackward-v1")
    metalearner.train(300)
    metalearner.save()