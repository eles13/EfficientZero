import numpy as np
from core.game import Game
from core.utils import arr_to_str
from typing import Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from pydantic import BaseModel

from pogema import GridConfig
from collections import deque


class PogemaWrapper(Game):
    def __init__(self, env, gc, discount: float, cvt_string=False):
        """True Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        discount: float
            discount of env
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        """
        super().__init__(env, env.action_space.n, discount)
        self.cvt_string = cvt_string
        self.gc = gc
        self.structs = deque()
        self.actions = []

    def legal_actions(self):
        return list(range(5))

    def get_max_episode_steps(self):
        return self.gc.max_episode_steps

    def step(self, action):
        if len(self.structs) > 0:
            current = self.structs.popleft()
            observation = current['observation']
            reward = current['reward']
            done = current['done']
            info = current['info']
            self.actions.append(action)
        else:
            self.actions.append(action)
            print(self.actions)
            observation, reward, done, info = self.env.step(self.actions[:self.gc.num_agents])
            self.structs = deque()
            for o, r, d, i in zip(observation[1:], reward[1:], done[1:], info[1:]):
                self.structs.append({'observation': o, 'reward': r, 'done': d, 'info': i})
            observation = observation[0]
            reward = reward[0]
            done = done[0]
            info = info[0]
            self.actions = []
        #observation = np.concatenate(observation, axis=-1)
        observation = observation.transpose(2,1,0)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        #observation = np.concatenate(observation, axis=-1)
        self.structs = deque()
        for o in observation[1:]:
            self.structs.append({'observation': o, 'reward': 0, 'done': False, 'info': {}})
        observation = observation[0].transpose(2,1,0)
        observation = observation.astype(np.uint8)
        self.actions = []

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation

    def close(self):
        self.env.close()


class POMAPFConfig(GridConfig):
    integration: Literal['SampleFactory'] = 'SampleFactory'
    collision_system: Literal['block_both'] = 'block_both'
    observation_type: Literal['POMAPF', 'MAPF'] = 'POMAPF'


class Environment(BaseModel, ):
    grid_config: POMAPFConfig = POMAPFConfig()
    name: str = "POMAPF-v0"
    grid_memory_obs_radius: Optional[int] = None
    observation_type: str = 'POMAPF'  # valid values - POMAPF, MAPF, FOG_OF_WAR
    sub_goal_distance: Optional[int] = None
