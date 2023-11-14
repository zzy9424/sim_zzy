import copy
import random
import sys
from collections import deque

from replays.CER import CER
from replays.EPER import EPER
from replays.default_replay import DefaultReplay
from replays.online import OnlineReplay
sys.path.append(".")
sys.path.append("minigrid")
import numpy as np
import torch
import torch.nn as nn

from minigrid.wrappers import *
from drl_lib import dqn
import argparse
import queue
import threading
import time

import tensorboard
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from replays.default_replay import DefaultReplay
from replays.proportional_PER.proportional import ProportionalPER
from replays.rank_PER.rank_based import RankPER

from distutils.util import strtobool

def train(args):
    acc_reward = 0
    # random_q= deque([0.92, 0.3, -0.16, -0.3, 0.22, -0.46, 0.47, -0.54, 0.17, -0.31, 0.87, -0.51, 0.04, -0.97, 0.94, -0.29, 0.77, -0.54, 0.23, -0.66])
    random_q = deque(
        [0.92, -0.83, 0.36, -0.16, 0.46, -0.65, 0.77, -0.54, 0.17, -0.31, 0.87, -0.51, 0.04, -0.97, 0.94, -0.29, 0.77,
         -0.54, 0.23, -0.66])

    acc_reward = 0
    env_name = args.env_name
    number = args.number
    random_seed = args.random_seed
    batch_size = args.batch_size  # num of transitions sampled from replay buffer
    max_timestep = int(args.max_timestep)
    max_epi_timestep = int(args.max_epi_timestep)
    model_size = args.model_size
    full_obs = int(args.full_obs)

    train_freq = args.train_freq
    save_rate = args.save_rate  # save the check point per ? episode
    replay_name = args.replay
    replay_max_size = args.replay_max_size
    if args.note !="":
        note = "_" + args.note
    else:
        note = args.note
    if not torch.cuda.is_available():
        device = "cpu"
    change_env = int(args.change_env) == 1
    if change_env:
        change_timestep = args.change_timestep
        change_type = args.change_type

    beta_decay = not (replay_name == "default" or args.beta == args.beta_min)
    if beta_decay:
        beta_delta = (args.beta - args.beta_min)/args.beta_decay_step
    print("beta_decay: ",beta_decay)

    # save setting
    directory = f"./models/{env_name}/{number}"
    log_directory = f'./runs/{env_name}/{number}{note}'

    # 检查文件夹是否存在
    if os.path.exists(directory):
        # 如果存在，询问用户是否覆盖
        overwrite = input(f"{directory} already exists. Do you want to overwrite or continue it? (y/n/c)")
        if overwrite == "y":
            os.system(f"rm -rf {directory}")
            os.system(f"rm -rf './runs/{env_name}/{number}'*")
        elif overwrite == "c":
            pass
        else:
            print("Aborting operation.")
            exit()

    os.makedirs(directory)
    os.makedirs(log_directory)
    writer = SummaryWriter(log_dir=log_directory)

    env = ImgObsWrapper(gym.make(env_name))
    if full_obs == 1:
        env = ImgObsWrapper(FullyObsWrapper(gym.make(env_name)))
    else:
        env = ImgObsWrapper(gym.make(env_name))

    if max_epi_timestep == -1:
        max_epi_timestep = env.max_steps

    input_shape = np.array(env.observation_space.shape)
    input_shape = np.insert(input_shape, 0, input_shape[2])
    input_shape = input_shape[:3]
    action_n = len(env.used_actions)
    print(f"action_n:{action_n}")

    if model_size == "large":
        net = dqn.DQN_large(input_shape, action_n)
    elif model_size == "very_large":
        net = dqn.DQN_very_large(input_shape, action_n)
    elif model_size == "small":
        net = dqn.DQN(input_shape, action_n)
    agent = dqn.DQNAgent(net,args,writer)
    if args.restart != "":
        agent.load(args.restart)
        train = False
    else:
        train = True
    np.save(f"{directory}/args_num_{number}.npy", args)
    with open(f'{directory}/meta.txt', 'w') as f:
        sys.stdout = f
        print(args)
        print(net)
        sys.stdout = sys.__stdout__

    print(replay_name)
    if replay_name == "default":
        replay_buffer = DefaultReplay(replay_max_size,batch_size)
    elif replay_name == "proportional_PER" or replay_name == "PER":
        replay_buffer = ProportionalPER(replay_max_size,batch_size,alpha=args.alpha)

    elif replay_name == "CER":
        replay_buffer = CER(replay_max_size,batch_size)
    elif replay_name == "EPER":
        replay_buffer = EPER(replay_max_size,batch_size,alpha=args.alpha,beta=args.beta,lambda_init=args.lambda_init)
    elif replay_name == "online":
        replay_buffer = OnlineReplay(replay_max_size, batch_size)
    else:
        raise Exception(f"No replay type found: {replay_name}")
    replay_buffer.writer = writer
    print(type(replay_buffer))
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


    # training procedure:
    total_step = 0
    episode = 0
    slot_num = 0
    slot_step = 0
    while total_step < max_timestep:
        if change_env:
            if slot_step >= change_timestep:
                slot_step = 0
                slot_num += 1
                if change_type == "noisy_obs":
                    if full_obs == 1:
                        # random.seed(10)
                        env = NoisyImgObsWrapper(FullyObsWrapper(gym.make(env_name)),noise=random_q.popleft(),noise_amp=args.noise_amp)
                    else:
                        env = NoisyImgObsWrapper(gym.make(env_name))
                if change_type == "sticky_action":
                    if full_obs == 1:
                        env = StickyImgObsWrapper(FullyObsWrapper(gym.make(env_name)))
                    else:
                        env = StickyImgObsWrapper(gym.make(env_name))
                if change_type == "obstacle":
                    # agent.epsilon = 0.5
                    # agent.epsilon_delta = (0.5 - args.epsilon_min) / args.epsilon_decay_step
                    if full_obs == 1:
                        env = ObstacleImgObsWrapper(gym.make(env_name),n_obstacle =int(args.change_factor),full_obs=True,env_name=env_name,slot_num=slot_num)
                    else:
                        env = ObstacleImgObsWrapper(gym.make(env_name),n_obstacle = int(args.change_factor),env_name=env_name,slot_num=slot_num)
                # changed = True
                print("=" * 15, "Env changed", "=" * 15)
                # train = True
                if replay_name == "EPER":
                    replay_buffer.stage_1_to_2()
                    replay_buffer.update_saved_net(copy.deepcopy(agent.net))
        episode +=1
        ep_reward = ep_step = 0
        done = False
        epi_start_time = time.time()
        state = process_obs(env.reset()[0])
        while(not done and ep_step < max_epi_timestep):
            total_step += 1
            ep_step += 1
            slot_step +=1
            # select action and add exploration noise:
            action = agent.select_action(state)
            # take action in env:
            next_state, reward, done, info,_ = env.step(env.used_actions[action])
            next_state = process_obs(next_state)
            # TD error:
            replay_buffer.add((state, action, reward, next_state, float(done)),priority=replay_buffer.max_p,age=total_step)
            state = next_state
            ep_reward += reward

            # train agent
            if train and total_step % train_freq == 0 and replay_buffer.size > batch_size:
                agent.update(replay_buffer,total_step)
                if beta_decay:
                    replay_buffer.beta = max(replay_buffer.beta - beta_delta, args.beta_min)

        # logging updates:
        writer.add_scalar("reward", ep_reward, global_step=total_step)
        # if changed:
        #     acc_reward += ep_reward
        # save checkpoint:
        if episode % save_rate == 0:
            agent.save(directory, int(total_step / 1000))
        episode_time = time.time() - epi_start_time

        if episode % 1 == 0:
            print("Exp: {}\t"
                  "Episode: {}\t"
                  "Step: {}k\t"
                  "Slot Step: {}k\t"
                  "Reward: {}\t"
                  "Replay: {}/{} \t"
                  "Epsilon: {} \t"
                  "Epi_step: {} \t"
                  "Epi_Time: {} \t".format(args.note,episode, int(total_step / 1000), int(slot_step / 1000),
                                         round(ep_reward, 2),
                                         replay_buffer.size,int(args.replay_max_size),
                                         agent.epsilon,
                                         ep_step, episode_time))
    with open('log.txt', 'a') as f:
        f.write(f"{note}\t{acc_reward}\n")
def process_obs(obs):
    return torch.tensor(obs).float().permute(2, 0, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.register('type', 'boolean', strtobool)
    parser.add_argument('--env_name', type=str, default='MiniGrid-Dynamic-Obstacles-6x6-v0', help='environment name')
    parser.add_argument('--alg_name', type=str, default='DQN', help='alg name')
    parser.add_argument('--number', type=int, default=-1, help='number')
    parser.add_argument('--note', type=str, default="")
    parser.add_argument('--random_seed', type=int, default=1, help='random seed')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount for future rewards')
    parser.add_argument('--batch_size', type=int, default=512, help='num of transitions sampled from replay buffer')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epsilon_init', type=float, default=0.8)
    parser.add_argument('--epsilon_min', type=float, default=0.01)
    parser.add_argument('--epsilon_decay_step', type=float, default=1e6)
    parser.add_argument('--polyak', type=float, default=0.995, help='target agent update parameter (1-tau)')
    parser.add_argument('--max_timestep', type=int, default=1e7, help='max num of timestep')
    parser.add_argument('--max_epi_timestep', type=int, default=-1, help='max timesteps in one episode')
    parser.add_argument('--train_freq', type=int, default=1, help='train the agent per ? episode')
    parser.add_argument('--save_rate', type=int, default=10000, help='save the check point per ? episode')
    parser.add_argument('--replay', type=str, default="default", help='default/proportional_PER/EPER/CER/online')
    parser.add_argument('--replay_max_size', type=int, default=1e5, help='')
    parser.add_argument('--alpha', type=float, default=0.7, help='')
    parser.add_argument('--beta', type=float, default=0.7, help='')
    parser.add_argument('--lambda_init', type=float, default=0.0001, help='')
    parser.add_argument('--model_size', type=str, default="small", help='')
    parser.add_argument('--full_obs', type=int, default=1, help='1=True')
    parser.add_argument('--restart', type=str, default="", help='')
    parser.add_argument('--beta_min', type=float, default=0.7)
    parser.add_argument('--beta_decay_step', type=float, default=50000)
    # ================================
    parser.add_argument('--change_env', type=int, default=0, help='1=True')
    parser.add_argument('--change_timestep', type=int, default=5e6, help='')
    parser.add_argument('--change_type', type=str, default='noisy_obs', help='')
    parser.add_argument('--change_factor', type=float, default=-1, help='')
    parser.add_argument('--noise_amp', type=float, default=1)
    args = parser.parse_args()
    train(args)
