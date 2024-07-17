# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import safety_gymnasium
import argparse
import os
import json
from collections import deque
from safepo.common.env import make_sa_mujoco_env, make_ma_mujoco_env, make_ma_multi_goal_env, make_sa_isaac_env
from safepo.common.model import ActorVCritic
from safepo.utils.config import multi_agent_velocity_map, multi_agent_goal_tasks
import numpy as np
import joblib
import torch
from safety_gymnasium.wrappers import SafeAutoResetWrapper, SafeRescaleAction, SafeUnsqueeze
from safepo.common.wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ShareEnv, SafeNormalizeObservation, MultiGoalEnv
def eval_single_agent(eval_dir, eval_episodes):

    torch.set_num_threads(1)
    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))

    env_id = config['task'] if 'task' in config.keys() else config['env_name']
    print(config['task'])
    env_norms = os.listdir(eval_dir)
    env_norms = [env_norm for env_norm in env_norms if env_norm.endswith('.pkl')]
    final_norm_name = sorted(env_norms)[-1]

    model_dir = eval_dir + '/torch_save'
    models = os.listdir(model_dir)
    models = [model for model in models if model.endswith('.pt')]
    final_model_name = sorted(models)[-1]

    model_path = model_dir + '/' + final_model_name
    norm_path = eval_dir + '/' + final_norm_name
    show=False
    if show:
        env = safety_gymnasium.make(env_id,render_mode="human")
    else:
        env = safety_gymnasium.make(env_id)
    env.reset(seed=None)

    env = SafeAutoResetWrapper(env)
    env = SafeRescaleAction(env, -1, 1)
    env = SafeNormalizeObservation(env)
    eval_env = SafeUnsqueeze(env)
    obs_space = env.observation_space
    act_space = env.action_space
    print(act_space)

    model = ActorVCritic(
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            hidden_sizes=config['hidden_sizes'],
        )
    model.actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    if os.path.exists(norm_path):
        norm = joblib.load(open(norm_path, 'rb'))['Normalizer']
        eval_env.obs_rms = norm

    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)

    for _ in range(eval_episodes):
        eval_done = False
        eval_obs, _ = eval_env.reset()
        eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32)
        eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
        step=0
        while not eval_done:
            step+=1
            # print(step)
            with torch.no_grad():
                act, _, _, _ = model.step(
                    eval_obs, deterministic=True
                )
            act = act.detach().squeeze().cpu().numpy()
            eval_obs, reward, cost, terminated, truncated, info = eval_env.step(act)
            eval_obs = torch.as_tensor(
                eval_obs, dtype=torch.float32
            )
            eval_rew += reward[0]
            eval_cost += cost[0]
            eval_len += 1
            eval_done = terminated[0] or truncated[0]
        print(eval_rew)
        eval_rew_deque.append(eval_rew)
        eval_cost_deque.append(eval_cost)
        eval_len_deque.append(eval_len)

    return sum(eval_rew_deque) / len(eval_rew_deque), sum(eval_cost_deque) / len(eval_cost_deque)

if __name__ == '__main__':
    # benchmark_eval()
    reward,cost = eval_single_agent(
        "runs/trpo/SafetyRacecarGoal0-v0/trpo/seed-000-2024-07-12-13-33-29", 10)
    print(reward,cost)
