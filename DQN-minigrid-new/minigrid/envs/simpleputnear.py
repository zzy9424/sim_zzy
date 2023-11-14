from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key
from minigrid.minigrid_env import MiniGridEnv


class SimplePutNearEnv(MiniGridEnv):

    """
    ## Description

    The agent is instructed through a textual string to pick up an object and
    place it next to another object. This environment is easy to solve with two
    objects, but difficult to solve with more, as it involves both textual
    understanding and spatial reasoning involving multiple objects.

    ## Mission Space

    "put the {move_color} {move_type} near the {target_color} {target_type}"

    {move_color} and {target_color} can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {move_type} and {target_type} Can be "box", "ball" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Drop an object    |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

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

    1. The agent picks up the wrong object.
    2. The agent drop the correct object near the target.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    N: number of objects.

    - `MiniGrid-PutNear-6x6-N2-v0`
    - `MiniGrid-PutNear-8x8-N3-v0`

    """

    def __init__(self, size=6, numObjs=2, max_steps: int | None = None, **kwargs):
        self.size = size
        self.used_actions = [0,1,2,3,4]
        self.reach_ckp = False
        self.ckp_reward = 0
        self.numObjs = numObjs
        self.obj_types = ["ball"]
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[
                COLOR_NAMES,
                self.obj_types,
                COLOR_NAMES,
                self.obj_types,
            ],
        )

        if max_steps is None:
            max_steps = 5 * size

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
    def _gen_mission(
        move_color: str, move_type: str, target_color: str, target_type: str
    ):
        return f"put the {move_color} {move_type} near the {target_color} {target_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        objs = []
        objPos = []

        def near_obj(env, p1):
            for p2 in objPos:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    return True
            return False

        # Until we have generated all the objects
        while len(objs) < self.numObjs:
            objType = "ball"
            objColor = "red"

            obj = Key(objColor)

            pos = self.place_obj(obj, reject_fn=near_obj)

            objs.append((objType, objColor))
            objPos.append(pos)

        # Randomize the agent start position and orientation
        self.place_agent()

        # Choose a random object to be moved
        objIdx = self._rand_int(0, len(objs))
        self.move_type, self.moveColor = objs[objIdx]
        self.move_pos = objPos[objIdx]

        # Choose a target object (to put the first object next to)
        while True:
            targetIdx = self._rand_int(0, len(objs))
            if targetIdx != objIdx:
                break
        self.target_type, self.target_color = objs[targetIdx]
        self.target_pos = objPos[targetIdx]

        self.mission = "put the {} {} near the {} {}".format(
            self.moveColor,
            self.move_type,
            self.target_color,
            self.target_type,
        )

    def step(self, action):
        preCarrying = self.carrying

        obs, reward, terminated, truncated, info = super().step(action)

        u, v = self.dir_vec
        ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v)
        tx, ty = self.target_pos
        mx, my = self.move_pos

        if self.step_count == 1:
            self.reach_ckp = False

        if self.carrying and not self.reach_ckp:
            reward = self._reward()/2
            self.ckp_reward = reward
            self.reach_ckp = True

        # If successfully dropping an object near the target
        if action == self.actions.drop and preCarrying:
            if abs(ox - tx) <= 1 and abs(oy - ty) <= 1 or abs(ox - mx) <= 1 and abs(oy - my) <= 1:
                reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info
