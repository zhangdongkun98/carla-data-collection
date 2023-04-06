from rldev import Data
from rldev import Data as Experience
import carla_utils as cu
from carla_utils import carla, rl_template

import torch
import time
import copy

from . import scenarios
from .sensors import sensors_param_list
from .agents_master import AgentListMaster



class Env_v0(rl_template.EnvSingleAgent):
    scenario_cls = scenarios.ScenarioUnstructured
    agents_master_cls = AgentListMaster
    recorder_cls = rl_template.PseudoRecorder

    sensors_params = sensors_param_list

    decision_frequency = 3
    control_frequency = 39

    perception_range = 50.0
    

    @torch.no_grad()
    def _step_train(self, action):
        timestamp = str(time.time())
        self.time_step += 1

        ### state
        if self.time_step == 1:
            state = self.agents_master.perception(self.step_reset, self.time_step)
        else:
            state = self.state
        ### reward
        epoch_info = self._check_epoch()
        reward = self.reward_function.run_step(state, action, self.agents_master, epoch_info)
        epoch_done = epoch_info.done
        
        ### record
        self.recorder.record_agents(self.time_step, self.agents_master, epoch_info)
        self.recorder.record_experience(self.time_step, self.agents_master, action)
        
        ### callback
        self.on_episode_step(reward, epoch_info)

        ### step
        self.agents_master.run_step( action.action if isinstance(action, Data) else action )

        ### next_state
        next_state = self.agents_master.perception(self.step_reset, timestamp=-1)
        self.state = copy.copy(next_state)

        ### experience
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([epoch_done], dtype=torch.float32)
        if done == True:
            self.on_episode_end()
        experience = Experience(
            state=state, action=action, next_state=next_state, reward=reward,
            done=done, timestamp=timestamp,
        )
        return experience, epoch_done, epoch_info
    
    

    @property
    def settings(self):
        if self.mode == 'real':
            st = cu.default_settings(sync=False, render=True, dt=0.0)
        else:
            st = cu.default_settings(sync=True, render=False, dt=1/self.control_frequency)
            
        return st

    def _check_epoch(self):
        '''check if collision, timeout, success, dones'''
        timeouts = [a.check_timeout(self.time_tolerance) for a in self.agents_master.agents]
        # timeouts = [self.check_timeout()]
        epoch_done = timeouts[0]
        epoch_info = Data(done=epoch_done, t=timeouts[0])
        return epoch_info
