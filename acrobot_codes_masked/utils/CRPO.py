import torch.nn as nn
import torch.nn.functional as F
import collections
import copy
import torch
import gym
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import numpy as np
from utils.torch_utils import use_cuda, Tensor, Variable, ValueFunctionWrapper
import utils.math_utils as math_utils
import matplotlib.pyplot as plt
from copy import deepcopy

from utils.DQNSoftmax import *
from utils.DQNRegressor import *


class CRPO:
    def __init__(
            self,
            env,
            eps,
            input_size = 6,
            output_size = 3,
            policy = None,
            value_function = None,
            cost_value_function_1 = None,
            cost_value_function_2 = None,
            height = -0.5,
            direction = 0,
            value_function_lr = 1.0,
            gamma = 0.9,
            episodes = 10,
            length = 200,
            max_kl = 0.02,
            cg_damping = 0.006,
            cg_iters = 10,
            residual_tol = 1e-10,
            ent_coeff = 0.0,
            batch_size = 5100,
            limit_1 = 50,
            limit_2 = 50,
            tolerance = -0.5
        ):
        # length = length + np.random.normal(0,30)
        self.eps=eps/length
        self.H=length
        self.height = height
        self.env = env
        self.input_size = input_size
        self.output_size = output_size
        self.direction = direction
        ## Hyperparameter
        self.value_function_lr = value_function_lr
        self.gamma = gamma
        self.episodes = episodes #10
        self.length = length  #500
        self.max_kl = max_kl
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.residual_tol = residual_tol
        self.ent_coeff = ent_coeff
        self.batch_size = batch_size

        self.limit = -limit_1 # I: 50
        self.limit2 = -limit_2 # I: 50
        self.tolerance = tolerance  # accept range -10 to -0.5
        self.noise=np.random.uniform(low=-0.5,high=0.5)
        if policy is None:
            self.policy = DQNSoftmax(input_size, output_size)
        else:
            self.policy = deepcopy(policy)

        if value_function is None:
            self.value_function = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        else:
            self.value_function = deepcopy(value_function)

        if cost_value_function_1 is None:
            self.cost_value_function_1 = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        else:
            # print('cost_1 else')
            self.cost_value_function_1 = deepcopy(cost_value_function_1)

        if cost_value_function_2 is None:
            self.cost_value_function_2 = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        else:
            # print('cost_2 else')
            self.cost_value_function_2 = deepcopy(cost_value_function_2)

        
        # self.value_function = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        # self.cost_value_function_1 = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        # self.cost_value_function_2 = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)


    def sample_action_from_policy(self, observation):
        observation_tensor = Tensor(observation).unsqueeze(0)
        prob = self.policy(Variable(observation_tensor, requires_grad=True))*Tensor(np.array([1,2,3]))
        for cnt in range(len(prob[0,:])):
            prob[0,cnt]=min(max(prob[0,cnt],0.001), 1-0.001)
        # prob = np.max(np.vstack((prob,0.001*np.ones((len(prob),)))),axis=0)
        # prob = np.min(np.vstack((prob, (1-0.001)*np.ones((len(prob),)))),axis=0)
        prob /=sum(prob[0])
        m=self.masking(observation, prob)
        probabilities=prob*m
        probabilities /=sum(probabilities[0])
        # print('1',probabilities)
        try:
            action = probabilities.multinomial(1)
        except RuntimeError:
            # print(probabilities,m)  
            raise RuntimeError('probability: ', probabilities, 'm :', m)
        
        # print('2',probabilities)
        return action, probabilities,m, prob


    def masking(self, observation, probabilities):
        # for cnt,action in enumerate(actions):
            m=np.zeros((3,))
            if self.direction == 1:
                
                if observation[4] < 0.0 or observation[5] < 0.0:
                    for cnt in range(3):
                        if cnt==2:
                            m[cnt]=self.eps/probabilities[0,2]
                        else:
                            m[cnt]=(1-self.eps)/(1-probabilities[0,2])
                else:
                    m=np.ones((3,))
               
            else:
                if observation[4] > 0.0 or  observation[5] > 0.0:
                    for cnt in range(3):
                        if cnt==0:
                            m[cnt]=self.eps/probabilities[0,0]
                        else:
                            m[cnt]=(1-self.eps)/(1-probabilities[0,0])
                            
                else: 
                    m=np.ones((3,))
                    
            # if self.direction == 1:
                
            #     if observation[5] < 0.0:
            #         for cnt in range(3):
            #             if cnt==2:
            #                 m[cnt]=self.eps
            #             else:
            #                 m[cnt]=(1-self.eps)/(1-probabilities[0,2])
            #     else:
            #         m=np.ones((3,))
               
            # else:
            #     if observation[5] > 0.0:
            #         for cnt in range(3):
            #             if cnt==0:
            #                 m[cnt]=self.eps
            #             else:
            #                 m[cnt]=(1-self.eps)/(1-probabilities[0,0])
                            
            #     else: 
            #         m=np.ones((3,))
            return Tensor(m)
        

    def flatten(self, l):
        return [item for sublist in l for item in sublist]

    def reward_function(self, cos1, sin1, cos2, sin2):
        # return sin1*sin2-cos1*cos2-cos1
        if sin1*sin2-cos1*cos2-cos1 > self.height:
            return 1.0#+self.noise
        else:
            return sin1*sin2-cos1*cos2-cos1

    def constraint_I(self, theta1_dot, action):
        if self.direction == 1:
            if theta1_dot < 0.0 and action == 2:
                return -1.0
            else:
                return 0.0
        else:
            if theta1_dot > 0.0 and action == 0:
                return -1.0
            else:
                return 0.0


    def constraint_II(self, theta2_dot, action):
        if self.direction == 1:
            if theta2_dot < 0.0 and action == 2:
                return -1.0
            else:
                return 0.0
        else:
            if theta2_dot > 0.0 and action == 0:
                return -1.0
            else:
                return 0.0

    def sample_trajectories(self):
        paths = []
        episodes_so_far = 0
        entropy = 0

        while episodes_so_far < self.episodes:
            episodes_so_far += 1
            observations, actions, rewards, costs, costs2, action_distributions,violation = [], [], [], [], [], [],0
            observation = self.env.reset()
            length_so_far = 0
            done = False
            while length_so_far < self.length and not done:
                if done: observation = self.env.reset()
                observations.append(observation)
                action, action_dist,m,prob = self.sample_action_from_policy(observation)
                actions.append(action)
                action_distributions.append(action_dist)
                entropy += -(action_dist * action_dist.log()).sum()

                reward = self.reward_function(observation[0], observation[1], observation[2], observation[3])
                cost = self.constraint_I(observation[4], action)
                cost2 = self.constraint_II(observation[5], action)

                rewards.append(reward)
                
                if cost<0 or cost2<0:
                    violation=1
                    print('Constraint violation!', action, action_dist,prob,m,observation,cost,cost2,self.direction)
                # elif self.direction==1:
                #     if observation[4]<0.0 or observation[5]<0.0:
                #         print(action, action_dist,m,observation)
                costs.append(cost)
                costs2.append(cost2)

                # next step
                observation, _, done, _ = self.env.step(action[0, 0].item())  ## change I
                if violation==1:
                    done=True
                length_so_far += 1
            #print("episode: ", episodes_so_far, "length: ", length)

            path = {"observations": observations,
                    "actions": actions,
                    "rewards": rewards,
                    "costs": costs,
                    "costs2": costs2,
                    "violation": violation,
                    "action_distributions": action_distributions}
            paths.append(path)

        observations = self.flatten([path["observations"] for path in paths])
        discounted_rewards = self.flatten([math_utils.discount(path["rewards"], self.gamma) for path in paths])
        total_reward = sum(self.flatten([path["rewards"] for path in paths])) / self.episodes
        ## add for cost
        discounted_costs = self.flatten([math_utils.discount(path["costs"], self.gamma) for path in paths])
        total_cost = sum(self.flatten([path["costs"] for path in paths])) / self.episodes
        discounted_costs2 = self.flatten([math_utils.discount(path["costs2"], self.gamma) for path in paths])
        total_cost2 = sum(self.flatten([path["costs2"] for path in paths])) / self.episodes
        average_violations = sum([path["violation"] for path in paths]) / self.episodes
        actions = self.flatten([path["actions"] for path in paths])
        action_dists = self.flatten([path["action_distributions"] for path in paths])
        entropy = entropy / len(actions)
        discounted_total_reward=sum(discounted_rewards)/self.episodes
        return observations, np.asarray(discounted_rewards), discounted_total_reward, np.asarray(discounted_costs), total_cost, \
            np.asarray(discounted_costs2), total_cost2, actions, action_dists, entropy, average_violations

    def mean_kl_divergence(self, model, policy_model, observations):
        observations_tensor = torch.cat(
            [Variable(Tensor(observation)).unsqueeze(0) for observation in observations])
        actprob = model(observations_tensor).detach() + 1e-8
        old_actprob = policy_model(observations_tensor)
        return torch.sum(old_actprob * torch.log(old_actprob / actprob), 1).mean()

    def hessian_vector_product(self, vector, policy_model, observations):
        policy_model.zero_grad()
        mean_kl_div = self.mean_kl_divergence(policy_model, policy_model, observations)
        kl_grad = torch.autograd.grad(
            mean_kl_div, policy_model.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        grad_vector_product = torch.sum(kl_grad_vector * vector)
        grad_grad = torch.autograd.grad(
            grad_vector_product, policy_model.parameters())
        fisher_vector_product = torch.cat(
            [grad.contiguous().view(-1) for grad in grad_grad]).data
        return fisher_vector_product + (self.cg_damping * vector.data)

    def conjugate_gradient(self, policy_model, observations, b):
        p = b.clone().data
        r = b.clone().data
        x = np.zeros_like(b.data.cpu().numpy())
        rdotr = r.double().dot(r.double())
        for _ in range(self.cg_iters):
            z = self.hessian_vector_product(Variable(p), policy_model, observations).squeeze(0)
            v = rdotr / p.double().dot(z.double())
            # x += v * p.cpu().numpy()
            x += v.cpu().numpy() * p.cpu().numpy() # change II
            r -= v * z
            newrdotr = r.double().dot(r.double())
            mu = newrdotr / rdotr
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < self.residual_tol:
                break
        return x

    def surrogate_loss(self, theta, policy_model, observations, actions, advantage):
        new_model = copy.deepcopy(policy_model)
        vector_to_parameters(theta, new_model.parameters())
        observations_tensor = torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in observations])
        prob_new = new_model(observations_tensor).gather(1, torch.cat(actions)).data
        prob_old = policy_model(observations_tensor).gather(1, torch.cat(actions)).data + 1e-8
        return -torch.mean((prob_new / prob_old) * advantage)

    def linesearch(self, x, policy_model, observations, actions, advantage, fullstep, expected_improve_rate):
        accept_ratio = .1
        max_backtracks = 10
        fval = self.surrogate_loss(x, policy_model, observations, actions, advantage)
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            #   print("Search number {}...".format(_n_backtracks + 1))
            xnew = x.data.cpu().numpy() + stepfrac * fullstep
            newfval = self.surrogate_loss(Variable(torch.from_numpy(xnew)), policy_model, observations, actions, advantage)
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            if ratio > accept_ratio and actual_improve > 0:
                return Variable(torch.from_numpy(xnew))
        return x


    def step(self):
        # Generate rollout
        # print('before sample: ',self.cost_value_function_1,self.cost_value_function_2)
        self.all_observations, all_discounted_rewards, discounted_total_reward, all_discounted_costs, total_cost,\
        all_discounted_costs2, total_cost2, all_actions, all_action_dists, \
        entropy , average_violations= self.sample_trajectories()
        # print(average_violations)
        num_batches = len(all_actions) // self.batch_size + 1
        for batch_num in range(num_batches):
            # print(batch_num)
            # print("Processing batch number {}".format(batch_num+1))
            observations = self.all_observations[batch_num * self.batch_size:(batch_num+1)*self.batch_size]
            discounted_rewards = all_discounted_rewards[batch_num*self.batch_size:(batch_num+1)*self.batch_size]
            discounted_costs = all_discounted_costs[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
            discounted_costs2 = all_discounted_costs2[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
            actions = all_actions[batch_num * self.batch_size:(batch_num+1)*self.batch_size]
            action_dists = all_action_dists[batch_num * self.batch_size:(batch_num+1)*self.batch_size]

            # Calculate the advantage of each step by taking the actual discounted rewards seen
            # and subtracting the estimated value of each state
            baseline = self.value_function.predict(observations).data
            
            cost_baseline = self.cost_value_function_1.predict(observations).data
            cost_baseline2 = self.cost_value_function_2.predict(observations).data
            discounted_rewards_tensor = Tensor(discounted_rewards).unsqueeze(1)
            discounted_costs_tensor = Tensor(discounted_costs).unsqueeze(1)
            discounted_costs_tensor2 = Tensor(discounted_costs2).unsqueeze(1)
            advantage = discounted_rewards_tensor - baseline
            cost_advantage = discounted_costs_tensor - cost_baseline
            cost_advantage2 = discounted_costs_tensor2 - cost_baseline2

            # Normalize the advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            cost_advantage = (cost_advantage - cost_advantage.mean()) / (cost_advantage.std() + 1e-8)
            cost_advantage2 = (cost_advantage2 - cost_advantage2.mean()) / (cost_advantage2.std() + 1e-8)

            # Calculate the surrogate loss as the elementwise product of the advantage and the probability ratio of actions taken
            new_p = torch.cat(action_dists).gather(1, torch.cat(actions))
            old_p = new_p.detach() + 1e-8
            prob_ratio = new_p / old_p
            surrogate_loss = -torch.mean(prob_ratio * Variable(advantage)) - (self.ent_coeff * entropy)
            cost_surrogate_loss = -torch.mean(prob_ratio * Variable(cost_advantage)) - (self.ent_coeff * entropy)
            cost_surrogate_loss2 = -torch.mean(prob_ratio * Variable(cost_advantage2)) - (self.ent_coeff * entropy)

            # Calculate the gradient of the surrogate loss
            self.policy.zero_grad()

            # if total_cost <= self.limit + self.tolerance:
            #     cost_surrogate_loss.backward(retain_graph=True)
            # elif total_cost2 <= self.limit2 + self.tolerance:
            #     cost_surrogate_loss2.backward(retain_graph=True)
            # else:
            surrogate_loss.backward(retain_graph=True)

            policy_gradient = parameters_to_vector([v.grad for v in self.policy.parameters()]).squeeze(0)

            if policy_gradient.nonzero().size()[0]:
                # Use conjugate gradient algorithm to determine the step direction in theta space
                step_direction = self.conjugate_gradient(self.policy, observations, -policy_gradient)
                step_direction_variable = Variable(torch.from_numpy(step_direction))

                # Do line search to determine the stepsize of theta in the direction of step_direction
                shs = .5 * step_direction.dot(self.hessian_vector_product(step_direction_variable, self.policy, observations).cpu().numpy().T)
                lm = np.sqrt(shs / self.max_kl)
                fullstep = step_direction / lm
                #gdotstepdir = -policy_gradient.dot(step_direction_variable).data[0]
                gdotstepdir = -policy_gradient.dot(step_direction_variable).data.item()  # change III
                #gdotstepdir = gdotstepdir_mid.item()

                ## In the paper, CRPO uses the estimated value function (critic's return) to decide whether to update objective function or constraints.
                ## However, in the experiment the estimation build from the Monte Carlo rollout is good enough.

                if total_cost <= self.limit + self.tolerance:
                    theta = self.linesearch(parameters_to_vector(self.policy.parameters()), self.policy, observations, actions,
                                        cost_advantage, fullstep, gdotstepdir / lm)
                elif total_cost2 <= self.limit2 + self.tolerance:
                    theta = self.linesearch(parameters_to_vector(self.policy.parameters()), self.policy, observations, actions,
                                        cost_advantage2, fullstep, gdotstepdir / lm)
                else:
                    theta = self.linesearch(parameters_to_vector(self.policy.parameters()), self.policy, observations, actions,
                                        advantage, fullstep, gdotstepdir / lm)

                # Fit the estimated value function to the actual observed discounted rewards
                ev_before = math_utils.explained_variance_1d(baseline.squeeze(1).cpu().numpy(), discounted_rewards)
                cost_ev_before = math_utils.explained_variance_1d(cost_baseline.squeeze(1).cpu().numpy(), discounted_costs)
                cost_ev_before2 = math_utils.explained_variance_1d(cost_baseline2.squeeze(1).cpu().numpy(), discounted_costs2)
                self.value_function.zero_grad()
                value_fn_params = parameters_to_vector(self.value_function.parameters())
                self.cost_value_function_1.zero_grad()
                cost_value_fn_params = parameters_to_vector(self.cost_value_function_1.parameters())
                self.cost_value_function_2.zero_grad()
                cost_value_fn_params2 = parameters_to_vector(self.cost_value_function_2.parameters())

                self.value_function.fit(observations, Variable(discounted_rewards_tensor))
                self.cost_value_function_1.fit(observations, Variable(discounted_costs_tensor))
                self.cost_value_function_2.fit(observations, Variable(discounted_costs_tensor2))

                ev_after = math_utils.explained_variance_1d(
                    self.value_function.predict(observations).data.squeeze(1).cpu().numpy(), discounted_rewards)
                cost_ev_after = math_utils.explained_variance_1d(
                    self.cost_value_function_1.predict(observations).data.squeeze(1).cpu().numpy(), discounted_costs)
                cost_ev_after2 = math_utils.explained_variance_1d(
                    self.cost_value_function_2.predict(observations).data.squeeze(1).cpu().numpy(), discounted_costs2)

                if ev_after < ev_before or np.abs(ev_after) < 1e-4:
                    vector_to_parameters(value_fn_params, self.value_function.parameters())

                if cost_ev_after < cost_ev_before or np.abs(cost_ev_after) < 1e-4:
                    vector_to_parameters(cost_value_fn_params, self.cost_value_function_1.parameters())

                if cost_ev_after2 < cost_ev_before2 or np.abs(cost_ev_after2) < 1e-4:
                    vector_to_parameters(cost_value_fn_params2, self.cost_value_function_2.parameters())

                # Update parameters of policy model
                old_model = copy.deepcopy(self.policy)
                old_model.load_state_dict(self.policy.state_dict())
                if any(np.isnan(theta.data.cpu().numpy())):
                    print("NaN detected. Skipping update...")
                else:
                    vector_to_parameters(theta, self.policy.parameters())

                kl_old_new = self.mean_kl_divergence(old_model, self.policy, observations)
                diagnostics = collections.OrderedDict([('Total Reward', discounted_total_reward), ('Total Cost', -1*total_cost), ('Total Cost2', -1*total_cost2),
                                                       ('Average Violations', average_violations)
                                                        # ('KL Old New', kl_old_new.data.item()), ('Entropy', entropy.data.item()), ('EV Before', ev_before),
                                                        # ('EV After', ev_after)
                                                        ])
                # for key, value in diagnostics.items():
                #     print("{}: {}".format(key, value))

            else:
                print("Policy gradient is 0. Skipping update...")

        return discounted_total_reward, -total_cost, -total_cost2, average_violations