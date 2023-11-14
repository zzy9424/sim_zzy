from __future__ import annotations

import math
import operator
from functools import reduce
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import logger, spaces
from gymnasium.core import ObservationWrapper, ObsType, Wrapper

import minigrid
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.world_object import Goal, Lava, Wall
import random
from minigrid.core.world_object import Ball
import envs
class ReseedWrapper(Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.

    Example:
        >>> import minigrid
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ReseedWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _ = env.reset(seed=123)
        >>> [env.np_random.integers(10) for i in range(10)]
        [0, 6, 5, 0, 9, 2, 2, 1, 3, 1]
        >>> env = ReseedWrapper(env, seeds=[0, 1], seed_idx=0)
        >>> _, _ = env.reset()
        >>> [env.np_random.integers(10) for i in range(10)]
        [8, 6, 5, 2, 3, 0, 0, 0, 1, 8]
        >>> _, _ = env.reset()
        >>> [env.np_random.integers(10) for i in range(10)]
        [4, 5, 7, 9, 0, 1, 8, 9, 2, 3]
        >>> _, _ = env.reset()
        >>> [env.np_random.integers(10) for i in range(10)]
        [8, 6, 5, 2, 3, 0, 0, 0, 1, 8]
        >>> _, _ = env.reset()
        >>> [env.np_random.integers(10) for i in range(10)]
        [4, 5, 7, 9, 0, 1, 8, 9, 2, 3]
    """

    def __init__(self, env, seeds=(0,), seed_idx=0):
        """A wrapper that always regenerate an environment with the same set of seeds.

        Args:
            env: The environment to apply the wrapper
            seeds: A list of seed to be applied to the env
            seed_idx: Index of the initial seed in seeds
        """
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            logger.warn(
                "A seed has been passed to `ReseedWrapper.reset` which is ignored."
            )
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        return self.env.reset(seed=seed, options=options)


class ActionBonus(gym.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ActionBonus
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> env_bonus = ActionBonus(env)
        >>> _, _ = env_bonus.reset(seed=0)
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited (state,action) pairs.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, truncated, info


class PositionBonus(Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.

    Note:
        This wrapper was previously called ``StateBonus``.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import PositionBonus
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> env_bonus = PositionBonus(env)
        >>> obs, _ = env_bonus.reset(seed=0)
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        0.7071067811865475
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited positions.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = tuple(env.agent_pos)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, truncated, info


class ImgObsWrapper(ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = ImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (7, 7, 3)
    """

    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        return obs["image"]


class OneHotPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import OneHotPartialObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs["image"][0, :, :]
        array([[2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0]], dtype=uint8)
        >>> env = OneHotPartialObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs["image"][0, :, :]
        array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
              dtype=uint8)
    """

    def __init__(self, env, tile_size=8):
        """A wrapper that makes the image observation a one-hot encoding of a partially observable agent view.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space["image"].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        new_image_space = spaces.Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], num_bits), dtype="uint8"
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        img = obs["image"]
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype="uint8")

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {**obs, "image": out}


class RGBImgObsWrapper(ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import RGBImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![NoWrapper](../figures/lavacrossing_NoWrapper.png)
        >>> env = RGBImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![RGBImgObsWrapper](../figures/lavacrossing_RGBImgObsWrapper.png)
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img = self.get_frame(highlight=True, tile_size=self.tile_size)

        return {**obs, "image": rgb_img}


class RGBImgPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs["image"])  # doctest: +SKIP
        ![NoWrapper](../figures/lavacrossing_NoWrapper.png)
        >>> env_obs = RGBImgObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> plt.imshow(obs["image"])  # doctest: +SKIP
        ![RGBImgObsWrapper](../figures/lavacrossing_RGBImgObsWrapper.png)
        >>> env_obs = RGBImgPartialObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> plt.imshow(obs["image"])  # doctest: +SKIP
        ![RGBImgPartialObsWrapper](../figures/lavacrossing_RGBImgPartialObsWrapper.png)
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        # Rendering attributes for observations
        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces["image"].shape
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img_partial = self.get_frame(tile_size=self.tile_size, agent_pov=True)

        return {**obs, "image": rgb_img_partial}


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding instead of the agent view.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FullyObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = FullyObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (11, 11, 3)
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {**obs, "image": full_grid}


class DictObservationSpaceWrapper(ObservationWrapper):
    """
    Transforms the observation space (that has a textual component) to a fully numerical observation space,
    where the textual instructions are replaced by arrays representing the indices of each word in a fixed vocabulary.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import DictObservationSpaceWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['mission']
        'avoid the lava and get to the green goal square'
        >>> env_obs = DictObservationSpaceWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['mission'][:10]
        [19, 31, 17, 36, 20, 38, 31, 2, 15, 35]
    """

    def __init__(self, env, max_words_in_mission=50, word_dict=None):
        """
        max_words_in_mission is the length of the array to represent a mission, value 0 for missing words
        word_dict is a dictionary of words to use (keys=words, values=indices from 1 to < max_words_in_mission),
                  if None, use the Minigrid language
        """
        super().__init__(env)

        if word_dict is None:
            word_dict = self.get_minigrid_words()

        self.max_words_in_mission = max_words_in_mission
        self.word_dict = word_dict

        self.observation_space = spaces.Dict(
            {
                "image": env.observation_space["image"],
                "direction": spaces.Discrete(4),
                "mission": spaces.MultiDiscrete(
                    [len(self.word_dict.keys())] * max_words_in_mission
                ),
            }
        )

    @staticmethod
    def get_minigrid_words():
        colors = ["red", "green", "blue", "yellow", "purple", "grey"]
        objects = [
            "unseen",
            "empty",
            "wall",
            "floor",
            "box",
            "key",
            "ball",
            "door",
            "goal",
            "agent",
            "lava",
        ]

        verbs = [
            "pick",
            "avoid",
            "get",
            "find",
            "put",
            "use",
            "open",
            "go",
            "fetch",
            "reach",
            "unlock",
            "traverse",
        ]

        extra_words = [
            "up",
            "the",
            "a",
            "at",
            ",",
            "square",
            "and",
            "then",
            "to",
            "of",
            "rooms",
            "near",
            "opening",
            "must",
            "you",
            "matching",
            "end",
            "hallway",
            "object",
            "from",
            "room",
        ]

        all_words = colors + objects + verbs + extra_words
        assert len(all_words) == len(set(all_words))
        return {word: i for i, word in enumerate(all_words)}

    def string_to_indices(self, string, offset=1):
        """
        Convert a string to a list of indices.
        """
        indices = []
        # adding space before and after commas
        string = string.replace(",", " , ")
        for word in string.split():
            if word in self.word_dict.keys():
                indices.append(self.word_dict[word] + offset)
            else:
                raise ValueError(f"Unknown word: {word}")
        return indices

    def observation(self, obs):
        obs["mission"] = self.string_to_indices(obs["mission"])
        assert len(obs["mission"]) < self.max_words_in_mission
        obs["mission"] += [0] * (self.max_words_in_mission - len(obs["mission"]))

        return obs


class FlatObsWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FlatObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = FlatObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs.shape
        (2835,)
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 28

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.maxStrLen,),
            dtype="uint8",
        )

        self.cachedStr: str = None

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"]

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert (
                len(mission) <= self.maxStrLen
            ), f"mission string too long ({len(mission)} chars)"
            mission = mission.lower()

            strArray = np.zeros(
                shape=(self.maxStrLen, self.numCharCodes), dtype="float32"
            )

            for idx, ch in enumerate(mission):
                if ch >= "a" and ch <= "z":
                    chNo = ord(ch) - ord("a")
                elif ch == " ":
                    chNo = ord("z") - ord("a") + 1
                elif ch == ",":
                    chNo = ord("z") - ord("a") + 2
                else:
                    raise ValueError(
                        f"Character {ch} is not available in mission string."
                    )
                assert chNo < self.numCharCodes, "%s : %d" % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs


class ViewSizeWrapper(ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ViewSizeWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = ViewSizeWrapper(env, agent_view_size=5)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (5, 5, 3)
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8"
        )

        # Override the environment's observation spaceexit
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        return {**obs, "image": image}


class DirectionObsWrapper(ObservationWrapper):
    """
    Provides the slope/angular direction to the goal with the observations as modeled by (y2 - y2 )/( x2 - x1)
    type = {slope , angle}

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import DirectionObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = DirectionObsWrapper(env, type="slope")
        >>> obs, _ = env_obs.reset()
        >>> obs['goal_direction']
        1.0
    """

    def __init__(self, env, type="slope"):
        super().__init__(env)
        self.goal_position: tuple = None
        self.type = type

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset()

        if not self.goal_position:
            self.goal_position = [
                x for x, y in enumerate(self.grid.grid) if isinstance(y, Goal)
            ]
            # in case there are multiple goals , needs to be handled for other env types
            if len(self.goal_position) >= 1:
                self.goal_position = (
                    int(self.goal_position[0] / self.height),
                    self.goal_position[0] % self.width,
                )

        return self.observation(obs), info

    def observation(self, obs):
        slope = np.divide(
            self.goal_position[1] - self.agent_pos[1],
            self.goal_position[0] - self.agent_pos[0],
        )

        if self.type == "angle":
            obs["goal_direction"] = np.arctan(slope)
        else:
            obs["goal_direction"] = slope

        return obs


class SymbolicObsWrapper(ObservationWrapper):
    """
    Fully observable grid with a symbolic state representation.
    The symbol is a triple of (X, Y, IDX), where X and Y are
    the coordinates on the grid, and IDX is the id of the object.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import SymbolicObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = SymbolicObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (11, 11, 3)
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=max(OBJECT_TO_IDX.values()),
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        objects = np.array(
            [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
        )
        agent_pos = self.env.agent_pos
        ncol, nrow = self.width, self.height
        grid = np.mgrid[:ncol, :nrow]
        _objects = np.transpose(objects.reshape(1, nrow, ncol), (0, 2, 1))

        grid = np.concatenate([grid, _objects])
        grid = np.transpose(grid, (1, 2, 0))
        grid[agent_pos[0], agent_pos[1], 2] = OBJECT_TO_IDX["agent"]
        obs["image"] = grid

        return obs

class StickyActionWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.sticky_factor = 1/3

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        if random.random() < self.sticky_factor:
            return self.env.step(0)
        return self.env.step(action)


class SparseNoisyImgObsWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]


    def observation(self, obs):

        # 获取原始观测值
        obs = np.array(obs["image"])
        obs_shape = obs.shape
        # print(obs[0])
        print(obs_shape)
        indices = np.random.choice(obs.size, size=1, replace=False)
        obs.flat[indices] = np.random.randint(low=1, high=10, size=len(indices), dtype=np.uint8)

        # Reshape noise to match obs shape
        obs = obs.reshape(obs_shape)

        # print(obs[0])

        return obs
        # Generate noise matrix
        noise = np.random.multivariate_normal(self.mean, self.cov, size=obs.shape[:-1])

        # Add noise to observation
        noisy_obs = obs + noise

class NoisyImgObsWrapper(ObservationWrapper):
    def __init__(self, env, noise,noise_amp):

        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]
        self.noise = noise * noise_amp
        self.offset = np.full( self.observation_space.shape, self.noise, dtype=float)


        print(self.offset)
    def observation(self, obs):
        # 获取原始观测值
        obs = np.array(obs["image"])
        # Generate noise matrix
        # noise = np.random.multivariate_normal(self.mean, self.cov, size=obs.shape[:-1])


        # 添加噪声到观测值中
        noisy_obs = obs + self.offset + self.noise * random.random()
        return noisy_obs

class StickyImgObsWrapper(ObservationWrapper):

    def __init__(self, env,sticky_factor = 1/10):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]
        self.sticky_factor = sticky_factor

    def observation(self, obs):
        return obs["image"]

    def step(self, action):
        if random.random() < self.sticky_factor:
            action = 0
        print(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs["image"], reward, terminated, truncated, None

# class ObstacleImgObsWrapper(ObservationWrapper):
#
#     def __init__(self, env, n_obstacle = 1,full_obs=False):
#         super().__init__(env)
#         if full_obs:
#             new_image_space = spaces.Box(
#                 low=0,
#                 high=255,
#                 shape=(self.env.width, self.env.height, 3),  # number of cells
#                 dtype="uint8",
#             )
#
#             self.observation_space = spaces.Dict(
#                 {**self.observation_space.spaces, "image": new_image_space}
#             )
#         else:
#             self.observation_space = env.observation_space.spaces["image"]
#         self.n_obstacles = n_obstacle
#         self.full_obs = full_obs
#
#     def reset(self, **kwargs):
#         obs, _ = self.env.reset(**kwargs)
#         # obstacle_coords = [(4, 2)]
#         # for i in range(1):
#         #     x, y = obstacle_coords.pop(random.randint(0, len(obstacle_coords) - 1))
#         #     # Place obstacles
#         #     self.put_obj(Lava(),x,y)
#         # obs = self.gen_obs()
#         return self.observation(obs), _
#
#     def observation(self, obs):
#         if self.full_obs:
#             objects = np.array(
#                 [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
#             )
#             agent_pos = self.env.agent_pos
#             ncol, nrow = self.width, self.height
#             grid = np.mgrid[:ncol, :nrow]
#             _objects = np.transpose(objects.reshape(1, nrow, ncol), (0, 2, 1))
#
#             grid = np.concatenate([grid, _objects])
#             grid = np.transpose(grid, (1, 2, 0))
#             grid[agent_pos[0], agent_pos[1], 2] = OBJECT_TO_IDX["agent"]
#             obs["image"] = grid
#         return obs["image"]
#
#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         return self.observation(obs), reward, terminated, truncated, None
#

class ObstacleImgObsWrapper(ObservationWrapper):

    def __init__(self, env, n_obstacle = 1,full_obs=True,env_name=None,slot_num=0):
        super().__init__(env)
        self.slot_num=slot_num

        if env_name.find("DistShift") != -1:
            print("Distshift")
            self.obstacle_coords = [(2, 4), (3, 3), (4, 4),(5, 3),(6, 4)]
            self.obstacle_type = Lava()
        elif env_name.find("SimpleCrossing") != -1:
            print("SimpleCrossing")
            self.obstacle_coords = [(2, 2), (2, 3), (2, 4), (4, 2), (4, 3)]
            self.obstacle_type = Wall()
        elif env_name.find("LavaCrossing") != -1:
            print("LavaCrossing")
            self.obstacle_coords = [(2, 2), (2, 3), (2, 4), (4, 2), (4, 3)]
            self.obstacle_type = Lava()
        elif env_name.find("Empty") != -1:
            print("Empty")
            self.obstacle_coords = [(2, 2), (2, 3), (3, 2), (3, 3), (4, 4)]
            self.obstacle_type = Lava()
        elif env_name.find("LavaGapS6") != -1:
            print("LavaGapS6")
            self.obstacle_coords = [(2,2), (2, 3), (3, 2), (3, 3), (3, 4)]
            self.obstacle_type = Lava()
        elif env_name.find("Door") != -1:
            print("Door")
            self.obstacle_coords = [(2, 2), (2, 3), (3, 2), (3, 3), (4, 3)]
            self.obstacle_type = Lava()
        else:
            raise Exception(f"env not found:{env_name}")
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )
        self.observation_space = self.observation_space.spaces["image"]

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)

        for i in range(1):
            x, y = self.obstacle_coords[self.slot_num-1]
            if x == 0:
                break
            if self.is_safe_path(x,y):
                # Place obstacles
                self.put_obj(self.obstacle_type,x,y)
        obs = self.gen_obs()
        return self.observation(obs), _

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return full_grid

    # This is for crossing env
    def is_safe_path(self,i, j):
        env = self.unwrapped
        # if env.grid.get(i, j) == minigrid.core.world_object.Wall:
        #     return False
        count = 0
        neighbor = [(i+1,j),(i,j+1),(i-1,j),(i,j-1)]
        for cord in neighbor:
            tile = env.grid.get(cord[0],cord[1])
            if type(tile) == minigrid.core.world_object.Wall or type(tile) == minigrid.core.world_object.Lava:
                count+=1
            elif type(tile) == minigrid.core.world_object.Door:
                return False
        return count != 2
