from __future__ import annotations

import random

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key
from minigrid.minigrid_env import MiniGridEnv


class FetchEnv(MiniGridEnv):

    """
    ## Description

    This environment has multiple objects of assorted types and colors. The
    agent receives a textual string as part of its observation telling it which
    object to pick up. Picking up the wrong object terminates the episode with
    zero reward.

    ## Mission Space

    "{syntax} {color} {type}"

    {syntax} is one of the following: "get a", "go get a", "fetch a",
    "go fetch a", "you must fetch a".

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "key" or "ball".

    ## Action Space

    | Num | Name         | Action               |
    |-----|--------------|----------------------|
    | 0   | left         | Turn left            |
    | 1   | right        | Turn right           |
    | 2   | forward      | Move forward         |
    | 3   | pickup       | Pick up an object    |
    | 4   | drop         | Unused               |
    | 5   | toggle       | Unused               |
    | 6   | done         | Unused               |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the correct object.
    2. The agent picks up the wrong object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    N: number of objects to be generated.

    - `MiniGrid-Fetch-5x5-N2-v0`
    - `MiniGrid-Fetch-6x6-N2-v0`
    - `MiniGrid-Fetch-8x8-N3-v0`

    """

    def __init__(self, size=8, numObjs=3, max_steps: int | None = None, **kwargs):
        self.numObjs = numObjs
        self.used_actions = [0,1,2,3]
        self.targetColor = "red"
        self.targetType= "ball"
        self.pos_arr=[(2,2),(2,4),(4,2),(4,4)]
        self.agent_start_pos = (1,1)
        self.agent_start_dir = 1
        MISSION_SYNTAX = [
            "get a",
            "go get a",
            "fetch a",
            "go fetch a",
            "you must fetch a",
        ]
        self.size = size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[MISSION_SYNTAX, COLOR_NAMES, ["ball"]],
        )

        if max_steps is None:
            max_steps = 5 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(syntax: str, color: str, obj_type: str):
        return f"{syntax} {color} {obj_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)


        objs = []
        objPos = []
        random_pos = random.sample(self.pos_arr, 2)

        objType = "ball"

        obj = Ball("red")

        objs.append((objType, "red"))
        objPos.append(random_pos[0])
        self.put_obj(obj,random_pos[0][0],random_pos[0][1])

        obj = Ball("blue")
        objs.append((objType, "blue"))
        objPos.append(random_pos[1])
        self.put_obj(obj,random_pos[1][0],random_pos[1][1])

        print(objPos)
        # Randomize the agent start position and orientation
        # self.place_agent()
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Choose a random object to be picked up
        objIdx = 0
        self.targetType, self.target_color = objs[objIdx]
        self.target_pos = objPos[objIdx]

        descStr = f"{self.target_color} {self.targetType}"
        self.mission = "go to the %s" % descStr
        # print(self.mission)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.carrying:
            if (
                self.carrying.color == self.targetColor
                and self.carrying.type == self.targetType
            ):
                reward = self._reward()
                terminated = True
            else:
                reward = 0
                terminated = True

        return obs, reward, terminated, truncated, info
