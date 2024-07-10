# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Goal level 0."""
import math
import os
import pickle
import random

from safety_gymnasium.assets.geoms import *
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.assets.free_geoms import Vases
import safety_gymnasium
import yaml

class GoalLevel0(BaseTask):
    """An agent must navigate to a goal."""
    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-1, -1, 1, 1]
        self._add_geoms(Goal(keepout=0.305,locations=[(-2, -2)],size=0.1))
        with open('array.pkl', 'rb') as file:
            locs = pickle.load(file)
        walls = Walls(num=len(locs),locations=locs,size=0.1)
        self._add_geoms(walls)
        # coords =[((10, 5), (59, 9)), ((10, 10), (14, 14)), ((10, 30), (14, 114)), ((10, 125), (14, 139)), ((15, 35), (29, 39)), ((15, 85), (29, 89)), ((15, 110), (34, 114)), ((15, 130), (19, 139)), ((20, 60), (29, 64)), ((20, 90), (29, 94)), ((20, 115), (24, 124)), ((20, 135), (74, 139)), ((25, 105), (34, 114)), ((25, 130), (34, 139)), ((30, 120), (34, 124)), ((55, 10), (74, 14)), ((55, 30), (64, 39)), ((55, 55), (74, 59)), ((60, 45), (64, 49)), ((60, 85), (74, 89)), ((60, 110), (79, 114)), ((65, 35), (74, 39)), ((70, 15), (74, 59)), ((70, 90), (74, 139)), ((75, 5), (84, 9)), ((75, 105), (79, 124)), ((80, 10), (84, 14)), ((80, 120), (104, 124)), ((85, 20), (89, 24)), ((85, 35), (104, 39)), ((85, 55), (104, 59)), ((95, 10), (104, 14)), ((95, 105), (104, 114)), ((95, 115), (99, 124)), ((100, 5), (119, 9)), ((100, 15), (104, 59)), ((100, 85), (114, 89)), ((100, 90), (104, 114)), ((100, 130), (114, 134)), ((105, 110), (114, 114)), ((105, 135), (179, 139)), ((110, 40), (119, 49)), ((115, 10), (134, 14)), ((115, 30), (134, 39)), ((115, 55), (134, 59)), ((130, 5), (139, 9)), ((130, 15), (134, 59)), ((135, 45), (144, 49)), ((135, 75), (144, 84)), ((140, 35), (164, 39)), ((140, 40), (144, 49)), ((140, 85), (149, 89)), ((140, 100), (144, 104)), ((145, 5), (149, 14)), ((145, 30), (154, 39)), ((145, 55), (164, 59)), ((145, 80), (174, 84)), ((145, 110), (159, 114)), ((150, 45), (154, 49)), ((155, 85), (159, 94)), ((155, 100), (159, 114)), ((155, 130), (159, 139)), ((160, 40), (164, 59)), ((165, 60), (179, 64)), ((165, 85), (169, 89)), ((175, 50), (179, 69)), ((175, 85), (184, 89)), ((180, 65), (184, 94)), ((180, 130), (184, 134))]
        # walls = Walls(coords)
        # self._add_geoms(walls)
        # self.agent.locations = ((0, 0),)
        self.last_dist_goal = None
        # self._is_load_static_geoms = True
        self.goal_type_mapping = {}
        self.goal_locs = [(-1.7, -1.7),(0.1, -1.7),(1.7,-1.7)]
        for idx,loc in enumerate(self.goal_locs):
            loc_str = str(loc[0])+","+str(loc[1])
            self.goal_type_mapping[loc_str] = idx
    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward

    def specific_reset(self):
        pass

    def specific_pre_reset(self):
        self.agent.locations = ((2.0, 2.0),)
        self.agent.rot = math.pi

        self.goal.locations = (random.choice(self.goal_locs),)
    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size
