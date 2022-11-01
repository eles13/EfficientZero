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


class PogemaWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=True):
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

    def legal_actions(self):
        return list(range(5))

    def get_max_episode_steps(self):
        return self.env.get_max_episode_steps()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = observation.astype(np.uint8)

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
