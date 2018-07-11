import gym
import sys
import os
import time
import copy
import math
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import create_np_map as CNP

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0:[0.0,0.0,0.0], 1:[0.5,0.5,0.5], \
          2:[0.0,0.0,1.0], 3:[0.0,1.0,0.0], \
          4:[1.0,0.0,0.0], 6:[1.0,0.0,1.0], \
          7:[1.0,1.0,0.0]}

class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    num_env = 0

    def __init__(self,map_x,map_y,local_x,local_y,heading,altitude,width=20,height=20):
        self.map_volume = CNP.map_to_volume_dict(map_x,map_y,width,height)
        self.local_coordinates = [local_x,local_y]
        self.world_coordinates = [70,50]
        self.actions = list(range(15))
        self.heading = heading
        self.altitude = altitude
        self.action_space = spaces.Discrete(15)
        # put the drone in
        self.map_volume[altitude]['drone'][local_x, local_y] = 1.0

        self.actionvalue_heading_action = {
            0: {1:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                2:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                3:'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                4:'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                5:'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=3)',
                6:'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                7:'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                8:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)'},
            1: {1:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                2:'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                3:'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                4:'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                5:'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                6:'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                7:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                8:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)'},
            2: {1:'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                2:'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                3:'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                4:'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                5:'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                6:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                7:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                8:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)'},
            3: {1:'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                2:'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                3:'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                4:'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                5:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                6:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                7:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                8:'self.take_action(delta_alt=-1,delta_x=-0,delta_y=-1,new_heading=1)'},
            4: {1:'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                2:'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                3:'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                4:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                5:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                6:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                7:'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                8:'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)'},
            5: {1:'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)',
                2: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=-1, new_heading=8)',
                3: 'self.take_action(delta_alt=0, delta_x=0, delta_y=-1, new_heading=1)',
                4: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=2)',
                5: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=3)',
                6: 'self.take_action(delta_alt=0, delta_x=1, delta_y=1, new_heading=4)',
                7: 'self.take_action(delta_alt=0, delta_x=0, delta_y=1, new_heading=5)',
                8: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=1, new_heading=6)'},
            6: {1: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=-1, new_heading=8)',
                2: 'self.take_action(delta_alt=0, delta_x=0, delta_y=-1, new_heading=1)',
                3: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=2)',
                4: 'self.take_action(delta_alt=0, delta_x=1, delta_y=0, new_heading=3)',
                5: 'self.take_action(delta_alt=0, delta_x=1, delta_y=1, new_heading=4)',
                6: 'self.take_action(delta_alt=0, delta_x=0, delta_y=1, new_heading=5)',
                7: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=1, new_heading=6)',
                8: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)'},
            7: {1: 'self.take_action(delta_alt=0, delta_x=0, delta_y=-1, new_heading=1)',
                2: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=2)',
                3: 'self.take_action(delta_alt=0, delta_x=1, delta_y=0, new_heading=3)',
                4: 'self.take_action(delta_alt=0, delta_x=1, delta_y=1, new_heading=4)',
                5: 'self.take_action(delta_alt=0, delta_x=0, delta_y=1, new_heading=5)',
                6: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=1, new_heading=6)',
                7: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)',
                8: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=-1, new_heading=8)'},
            8: {1: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=2)',
                2: 'self.take_action(delta_alt=0, delta_x=1, delta_y=0, new_heading=3)',
                3: 'self.take_action(delta_alt=0, delta_x=1, delta_y=1, new_heading=4)',
                4: 'self.take_action(delta_alt=0, delta_x=0, delta_y=1, new_heading=5)',
                5: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=1, new_heading=6)',
                6: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)',
                7: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=-1, new_heading=8)',
                8: 'self.take_action(delta_alt=0, delta_x=-0, delta_y=-1, new_heading=1)'},
            9: {1: 'self.take_action(delta_alt=0, delta_x=1, delta_y=0, new_heading=3)',
                2: 'self.take_action(delta_alt=0, delta_x=1, delta_y=1, new_heading=4)',
                3: 'self.take_action(delta_alt=0, delta_x=0, delta_y=1, new_heading=5)',
                4: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=1, new_heading=6)',
                5: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)',
                6: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=-1, new_heading=8)',
                7: 'self.take_action(delta_alt=0, delta_x=0, delta_y=-1, new_heading=1)',
                8: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=2)'},
            10: {1: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=0, new_heading=7)',
                2: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=-1, new_heading=8)',
                3: 'self.take_action(delta_alt=1, delta_x=0, delta_y=-1, new_heading=1)',
                4: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=2)',
                5: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=3)',
                6: 'self.take_action(delta_alt=1, delta_x=1, delta_y=1, new_heading=4)',
                7: 'self.take_action(delta_alt=1, delta_x=0, delta_y=1, new_heading=5)',
                8: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=1, new_heading=6)'},
            11: {1: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=-1, new_heading=8)',
                2: 'self.take_action(delta_alt=1, delta_x=0, delta_y=-1, new_heading=1)',
                3: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=2)',
                4: 'self.take_action(delta_alt=1, delta_x=1, delta_y=0, new_heading=3)',
                5: 'self.take_action(delta_alt=1, delta_x=1, delta_y=1, new_heading=4)',
                6: 'self.take_action(delta_alt=1, delta_x=0, delta_y=1, new_heading=5)',
                7: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=1, new_heading=6)',
                8: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=0, new_heading=7)'},
            12: {1: 'self.take_action(delta_alt=1, delta_x=0, delta_y=-1, new_heading=1)',
                2: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=2)',
                3: 'self.take_action(delta_alt=1, delta_x=1, delta_y=0, new_heading=3)',
                4: 'self.take_action(delta_alt=1, delta_x=1, delta_y=1, new_heading=4)',
                5: 'self.take_action(delta_alt=1, delta_x=0, delta_y=1, new_heading=5)',
                6: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=1, new_heading=6)',
                7: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=0, new_heading=7)',
                8: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=-1, new_heading=8)'},
            13: {1: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=2)',
                2: 'self.take_action(delta_alt=1, delta_x=1, delta_y=0, new_heading=3)',
                3: 'self.take_action(delta_alt=1, delta_x=1, delta_y=1, new_heading=4)',
                4: 'self.take_action(delta_alt=1, delta_x=0, delta_y=1, new_heading=5)',
                5: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=1, new_heading=6)',
                6: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=0, new_heading=7)',
                7: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=-1, new_heading=8)',
                8: 'self.take_action(delta_alt=1, delta_x=-0, delta_y=-1, new_heading=1)'},
            14: {1: 'self.take_action(delta_alt=1, delta_x=1, delta_y=0, new_heading=3)',
                2: 'self.take_action(delta_alt=1, delta_x=1, delta_y=1, new_heading=4)',
                3: 'self.take_action(delta_alt=1, delta_x=0, delta_y=1, new_heading=5)',
                4: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=1, new_heading=6)',
                5: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=0, new_heading=7)',
                6: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=-1, new_heading=8)',
                7: 'self.take_action(delta_alt=1, delta_x=0, delta_y=-1, new_heading=1)',
                8: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=2)'}
        


        }


        print("here")


        # self.action_map = {
        #     0: self.take_action(self.heading,self.altitude,-1,self.heading-1,self.heading-2), #turn left, down 1
        #
        # }

    # def __init__(self):
    #     self.actions = list(range(15))
    #     self.inv_actions = [0, 2, 1, 4, 3]
    #     self.action_space = spaces.Discrete(15)
    #     self.action_pos_dict = {0: [0,0], 1:[-1, 0], 2:[1,0], 3:[0,-1], 4:[0,1]}
    #
    #     ''' set observation space '''
    #     self.obs_shape = [128, 128, 3]  # observation space shape
    #     self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape)
    #
    #     ''' initialize system state '''
    #     this_file_path = os.path.dirname(os.path.realpath(__file__))
    #     self.grid_map_path = os.path.join(this_file_path, 'plan5.txt')
    #     self.start_grid_map = self._read_grid_map(self.grid_map_path) # initial grid map
    #     self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
    #     self.observation = self._gridmap_to_observation(self.start_grid_map)
    #     self.grid_map_shape = self.start_grid_map.shape
    #
    #     ''' agent state: start, target, current state '''
    #     self.agent_start_state, _ = self._get_agent_start_target_state(
    #                                 self.start_grid_map)
    #     _, self.agent_target_state = self._get_agent_start_target_state(
    #                                 self.start_grid_map)
    #     self.agent_state = copy.deepcopy(self.agent_start_state)
    #
    #     ''' set other parameters '''
    #     self.restart_once_done = False  # restart or not once done
    #     self.verbose = False # to show the environment or not
    #
    #     GridworldEnv.num_env += 1
    #     self.this_fig_num = GridworldEnv.num_env
    #     if self.verbose == True:
    #         self.fig = plt.figure(self.this_fig_num)
    #         plt.show(block=False)
    #         plt.axis('off')
    #         self._render()

    def take_action(self,delta_alt=0,delta_x=0,delta_y=0,new_heading=1):
        print("take heading called",delta_alt,delta_x,delta_y,new_heading)
        self.map_volume[self.altitude]['drone'][self.local_coordinates[0],self.local_coordinates[1]] = 0.0
        self.map_volume[self.altitude + delta_alt]['drone'][self.local_coordinates[0]+delta_x,self.local_coordinates[1]+delta_y] = 1.0
        self.altitude += delta_alt


    def check_for_crash(self):
        #if drone on altitude 0, crash
        if self.map_volume[0]['drone'].nonzero():
            return 1
        #at any other altutidue, check for an object at the drone's position
        drone_position = self.map_volume[self.altitude]['drone'].nonzero()
        for i in range(self.altitude,4):

            for key in self.map_volume[i]:
                if key == 'drone':
                    continue
                hazards = self.map_volume[i][key].nonzero()
                if drone_position in hazards:
                    return 1
        return 0




    def _step(self, action):
        ''' return next observation, reward, finished, success '''

        action = int(action)
        eval(self.actionvalue_heading_action[action][self.heading])
        crash = self.check_for_crash()

        return (self.map_volume, 0, True, crash)



        return 0



        # action = int(action)
        # info = {}
        # info['success'] = False
        # nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
        #                     self.agent_state[1] + self.action_pos_dict[action][1])
        # if action == 0: # stay in place
        #     info['success'] = True
        #     return (self.observation, 0, False, info)
        # if nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]:
        #     info['success'] = False
        #     return (self.observation, 0, False, info)
        # if nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1]:
        #     info['success'] = False
        #     return (self.observation, 0, False, info)
        # # successful behavior
        # org_color = self.current_grid_map[self.agent_state[0], self.agent_state[1]]
        # new_color = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]
        # if new_color == 0:
        #     if org_color == 4:
        #         self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
        #         self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
        #     elif org_color == 6 or org_color == 7:
        #         self.current_grid_map[self.agent_state[0], self.agent_state[1]] = org_color-4
        #         self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
        #     self.agent_state = copy.deepcopy(nxt_agent_state)
        # elif new_color == 1: # gray
        #     info['success'] = False
        #     return (self.observation, 0, False, info)
        # elif new_color == 2 or new_color == 3:
        #     self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
        #     self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = new_color+4
        #     self.agent_state = copy.deepcopy(nxt_agent_state)
        # self.observation = self._gridmap_to_observation(self.current_grid_map)
        # self._render()
        # if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1] :
        #     target_observation = copy.deepcopy(self.observation)
        #     if self.restart_once_done:
        #         self.observation = self._reset()
        #         info['success'] = True
        #         return (self.observation, 1, True, info)
        #     else:
        #         info['success'] = True
        #         return (target_observation, 1, True, info)
        # else:
        #     info['success'] = True
        #     return (self.observation, 0, False, info)

    def _reset(self):
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self._render()
        return self.observation

    def _read_grid_map(self, grid_map_path):
        grid_map = open(grid_map_path, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array)
        return grid_map_array

    def _get_agent_start_target_state(self, start_grid_map):
        start_state = None
        target_state = None
        for i in range(start_grid_map.shape[0]):
            for j in range(start_grid_map.shape[1]):
                this_value = start_grid_map[i,j]
                if this_value == 4:
                    start_state = [i,j]
                if this_value == 3:
                    target_state = [i,j]
        if start_state is None or target_state is None:
            sys.exit('Start or target state not specified')
        return start_state, target_state

    def _gridmap_to_observation(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.random.randn(*obs_shape)*0.0
        gs0 = int(observation.shape[0]/grid_map.shape[0])
        gs1 = int(observation.shape[1]/grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                for k in range(3):
                    this_value = COLORS[grid_map[i,j]][k]
                    observation[i*gs0:(i+1)*gs0, j*gs1:(j+1)*gs1, k] = this_value
        return observation
  
    def _render(self, mode='human', close=False):
        return
        if self.verbose == False:
            return
        img = self.observation
        fig = plt.figure(self.this_fig_num)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.00001)
        return 
 
    def change_start_state(self, sp):
        ''' change agent start state '''
        ''' Input: sp: new start state '''
        if self.agent_start_state[0] == sp[0] and self.agent_start_state[1] == sp[1]:
            _ = self._reset()
            return True
        elif self.start_grid_map[sp[0], sp[1]] != 0:
            return False
        else:
            s_pos = copy.deepcopy(self.agent_start_state)
            self.start_grid_map[s_pos[0], s_pos[1]] = 0
            self.start_grid_map[sp[0], sp[1]] = 4
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_start_state = [sp[0], sp[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self._reset()
            self._render()
        return True
        
    
    def change_target_state(self, tg):
        if self.agent_target_state[0] == tg[0] and self.agent_target_state[1] == tg[1]:
            _ = self._reset()
            return True
        elif self.start_grid_map[tg[0], tg[1]] != 0:
            return False
        else:
            t_pos = copy.deepcopy(self.agent_target_state)
            self.start_grid_map[t_pos[0], t_pos[1]] = 0
            self.start_grid_map[tg[0], tg[1]] = 3
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_target_state = [tg[0], tg[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self._reset()
            self._render()
        return True
    
    def get_agent_state(self):
        ''' get current agent state '''
        return self.agent_state

    def get_start_state(self):
        ''' get current start state '''
        return self.agent_start_state

    def get_target_state(self):
        ''' get current target state '''
        return self.agent_target_state

    def _jump_to_state(self, to_state):
        ''' move agent to another state '''
        info = {}
        info['success'] = True
        if self.current_grid_map[to_state[0], to_state[1]] == 0:
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 4:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, info)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 6:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 2
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, info)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 7:  
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 3
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 4:
            return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 1:
            info['success'] = False
            return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 3:
            self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
            self.current_grid_map[to_state[0], to_state[1]] = 7
            self.agent_state = [to_state[0], to_state[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self._render()
            if self.restart_once_done:
                self.observation = self._reset()
                return (self.observation, 1, True, info)
            return (self.observation, 1, True, info)
        else:
            info['success'] = False
            return (self.observation, 0, False, info)

    def _close_env(self):
        plt.close(1)
        return
    
    def jump_to_state(self, to_state):
        a, b, c, d = self._jump_to_state(to_state)
        return (a, b, c, d) 


a = GridworldEnv(70,50,1,1,1,2)

print('complete')