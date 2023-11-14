import os.path
import pickle
import re
import time

import pandas as pd
import torch
import numpy as np
from gym import wrappers
from torch.utils.tensorboard import SummaryWriter
import argparse
from replay_buffer import ReplayBuffer
from maddpg import MADDPG
from matd3 import MATD3
import copy
import gym
import rsoccer_gym


class Runner:
    def __init__(self, args, env_name, number, seed, note):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.evaluate = self.args.evaluate_freq > 0
        print("evaluate:",self.evaluate)
        # Create env
        self.env = gym.make(self.env_name)
        # self.env_evaluate = gym.make(self.env_name)
        self.args.N = 3  # The number of agents
        self.args.obs_dim_n = [self.env.observation_space.shape[1] for i in range(self.args.N)]
        self.args.action_dim_n = [self.env.action_space.shape[1] for i in range(self.args.N)]
        # self.args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in range(self.args.N)]  # obs dimensions of N agents
        # self.args.action_dim_n = [self.env.action_space[i].shape[0] for i in range(self.args.N)]  # actions dimensions of N agents
        self.evaluate_df = pd.DataFrame(columns=["episode_ckp","goal_diff_0","goal_diff_1","goal_diff_2","goal_diff_3","goal_diff_4"])
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/{}/{}_{}_{}'.format(self.args.algorithm, self.env_name, self.number, note))

        # Create N agents
        if self.args.algorithm == "MADDPG":
            print("Algorithm: MADDPG")
            self.agent_n = [MADDPG(args, agent_id) for agent_id in range(args.N)]
        elif self.args.algorithm == "MATD3":
            print("Algorithm: MATD3")
            self.agent_n = [MATD3(args, agent_id,self.writer) for agent_id in range(args.N)]
        else:
            print("Wrong!!!")

        self.replay_buffer = ReplayBuffer(self.args)

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.episode = 0
        if self.args.display:
            self.noise_std = 0
        else:
            self.noise_std = self.args.noise_std_init  # Initialize noise_std


    def run(self, ):
        while self.episode < self.args.max_episode:
            epi_start_time = time.time()
            obs_n = self.env.reset()
            terminate = False
            done = False
            episode_step = 0
            episode_reward = 0
            while not (done or terminate):
                # Each agent selects actions based on its own local observations(add noise for exploration)
                a_n = [agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, obs_n)]
                # --------------------------!!!注意！！！这里一定要deepcopy，MPE环境会把a_n乘5-------------------------------------------
                obs_next_n, r_n, done, info = self.env.step(copy.deepcopy(a_n))
                if self.args.display:
                    self.env.render()
                # Store the transition
                self.replay_buffer.store_transition(obs_n, a_n, r_n, obs_next_n, done)
                obs_n = obs_next_n
                self.total_steps += 1
                episode_step += 1
                episode_reward += sum(r_n.values())
                # Decay noise_std
                if self.args.use_noise_decay:
                    self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                if episode_step >= self.args.episode_limit:
                    terminate = True

                train_start_time = time.time()
                if self.replay_buffer.current_size > self.args.batch_size and not self.args.display:
                    # Train each agent individually
                    for agent_id in range(self.args.N):
                        self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)

            end_time = time.time()
            train_time = end_time - train_start_time
            epi_time = end_time - epi_start_time
            self.episode += 1
            avg_train_reward=episode_reward/episode_step
            print("EPI: {}\t\tTOTAL_STEP: {}\t\tEPI_STEP: {}\t\tREWARD: {:.4f}\t\tGOAL: {}\t\tNOISE: {:.4f}\t\tEPI_TIME:{:.4f}\t\tTRAIN_TIME:{:.4f}".
                  format(self.episode,self.total_steps,episode_step,avg_train_reward,info["goal_score"],self.noise_std,epi_time,train_time))

            if not self.args.display and self.writer is not None:
                self.writer.add_scalar('Agent rewards for each episode',avg_train_reward,global_step=self.episode)
                self.writer.add_scalar('Goal', info["goal_score"], global_step=self.episode)

            if self.evaluate and self.episode % self.args.evaluate_freq == 0:
                goal_diff=self.evaluate_policy()
                self.evaluate_df.loc[len(self.evaluate_df.index)] = [self.episode,goal_diff[0],goal_diff[1],goal_diff[2],goal_diff[3],goal_diff[4]]

            if self.episode % self.args.save_rate == 0:
                for agent_id in range(self.args.N):
                    torch.save(self.agent_n[agent_id].actor.state_dict(),f"models/{number}/{self.episode}_agent_{agent_id}")

        self.env.close()
        # self.env_evaluate.close()
        self.evaluate_df.to_csv(f"./matd3_conv_result_5000.csv", index=False)


    def evaluate_policy(self, ):
        goal_diff=[0,0,0,0,0]
        for test_time in range(5):
            for _ in range(self.args.evaluate_times):
                match_step = 0
                while match_step < 5000:
                    obs_n = self.env.reset()
                    episode_step = 0
                    while True:
                        a_n = [agent.choose_action(obs, noise_std=0) for agent, obs in zip(self.agent_n, obs_n)]  # We do not add noise when evaluating
                        obs_next_n, r_n, done_n, info = self.env.step(copy.deepcopy(a_n))
                        goal_diff[test_time] += info["goal_score"]
                        episode_step += 1
                        obs_n = obs_next_n
                        match_step += 1
                        if done_n or match_step >= 5000:
                            break
        return goal_diff

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_episode", type=int, default=int(1e10), help=" Maximum number of episode")
    parser.add_argument("--episode_limit", type=int, default=300, help="Maximum number of steps per episode")#300
    parser.add_argument("--save_rate", type=float, default=100, help="Save every 'save_rate' episode")

    parser.add_argument("--evaluate_freq", type=float, default=-1, help="Evaluate the policy every 'evaluate_freq' episode, -1 for not evaluate")
    parser.add_argument("--evaluate_times", type=float, default=10, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=1, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=5e6, help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")
    parser.add_argument("--restore", type=bool, default=False, help="Restore from checkpoint")
    parser.add_argument("--restore_model_dir", type=str, default="../../HRL/models/matd3_adv/actor_number_23_1589k_agent_{}.pth", help="Restore from checkpoint")
    parser.add_argument("--display", type=bool, default=False, help="Display mode")
    parser.add_argument("--record", type=bool, default=False, help="Save evaluate video")
    parser.add_argument("--record_dir", type=str, default='./videos', help="Video save path")
    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps


    env_name = "VSSMA-v0"
    seed = 0
    number = 4
    note = "train_each_step"

    os.makedirs(f"models/{number}", exist_ok=True)
    with open(f"models/{number}/args.npy", 'wb') as f:
        pickle.dump(args, f)

    runner = Runner(args, env_name=env_name, number=number, seed=seed, note=note)
    with open(f'./models/{number}/args.npy', 'wb+') as f:
        pickle.dump(runner.args, f)

    if args.restore:
        load_number = re.findall(r"number_(.+?)_", args.restore_model_dir)[0]
        # assert load_number == str(number)
        print("Loading...")
        for i in range(len(runner.agent_n)):
            runner.agent_n[i].actor.load_state_dict(torch.load(args.restore_model_dir.format(i)))
            print(f"Load ckp:{args.restore_model_dir.format(i)}")
        try:
            with open('./runner/{}_env_{}_number_{}.pkl'.format(args.algorithm, env_name, number), 'rb') as f:
                runner.total_steps,runner.episode,runner.noise_std,runner.evaluate_rewards,runner.replay_buffer = pickle.load(f)
        except:
            pass
    print("start runner.run()")
    runner.run()
