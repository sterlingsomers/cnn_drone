import gym
import sys
import os
import time
import copy
import math
import itertools
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import threading
import random
import pygame
from scipy.misc import imresize

from gym_gridworld.envs import create_np_map as CNP

#from mavsim_server import MavsimHandler

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0: [0.0, 0.0, 0.0], 1: [0.5, 0.5, 0.5], \
          2: [0.0, 0.0, 1.0], 3: [0.0, 1.0, 0.0], \
          4: [1.0, 0.0, 0.0], 6: [1.0, 0.0, 1.0], \
          7: [1.0, 1.0, 0.0]}


class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    num_envs = 1

    def __init__(self, map_x=0, map_y=0, local_x=0, local_y=0, heading=1, altitude=2, hiker_x=5, hiker_y=5, width=20,
                 height=20):

        # # TODO: Pass the environment with arguments

        #num_alts = 4
        self.verbose = False # to show the environment or not
        self.dropping = True # This is for the reset to select the proper starting locations for hiker and drone
        self.restart_once_done = True  # restart or not once done
        self.drop = False
        self.maps = [(390,50)]#[(400,35), (350,90), (430,110),(390,50), (230,70)] #[(86, 266)] (70,50) # For testing, 70,50 there is no where to drop in the whole map
        self.dist_old = 1000
        self.drop_package_grid_size_by_alt = {1: 3, 2: 5, 3: 7}
        self.factor = 5
        self.reward = 0
        self.action_space = spaces.Discrete(16)
        self.actions = list(range(self.action_space.n))
        self.obs_shape = [50,50,3]
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape)
        self.real_actions = False

        if self.real_actions:
            self.mavsimhandler = MavsimHandler()
            stateThread = threading.Thread(target=self.mavsimhandler.read_state)
            stateThread.start()

        self.image_layers = {}

        # 5x5 plane descriptions
        self.planes = {}
        self.planes[1] = [[(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)], np.zeros((5, 5, 3))]
        self.planes[2] = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 4), (1, 3), (2, 3), (1, 2)], np.zeros((5, 5, 3))]
        self.planes[3] = [[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (1, 3), (2, 3), (3, 3), (2, 4)], np.zeros((5, 5, 3))]
        self.planes[4] = [[(0, 4), (1, 3), (2, 3), (3, 3), (4, 4), (2, 2), (3, 2), (3, 1), (4, 0)], np.zeros((5, 5, 3))]
        self.planes[5] = [[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (4, 2)], np.zeros((5, 5, 3))]
        self.planes[6] = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (2, 1), (3, 1), (3, 2), (4, 0)], np.zeros((5, 5, 3))]
        self.planes[7] = [[(2, 0), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)], np.zeros((5, 5, 3))]
        self.planes[8] = [[(0, 0), (4, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (1, 3), (0, 4)], np.zeros((5, 5, 3))]

        self.hikers = {}
        self.hikers[0] = [[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
                          np.zeros((5, 5, 3))]
        self.hiker_image = np.zeros((5, 5, 3))
        # self.hiker_image[:,:,:] = self.map_volume['feature_value_map']['hiker']['color']

        self.drop_probabilities = {"damage_probability": {0: 0.00, 1: 0.01, 2: 0.40, 3: 0.80},
                                   "stuck_probability": {"pine trees": 0.50, "pine tree": 0.25, "cabin": 0.50,
                                                         "flight tower": 0.15, "firewatch tower": 0.20},
                                   "sunk_probability": {"water": 0.50}
                                   }
        self.drop_rewards = {"OK": 1,#10,
                             # "OK_STUCK": 5,
                             # "OK_SUNK": 5,
                             "DAMAGED": -1,#-10,
                             # "DAMAGED_STUCK": -15,
                             # "DAMAGED_SUNK": -15,
                             # "CRASHED": -30
                             }
        self.alt_rewards = {0:-1, 1:1, 2:-0.5, 3:-0.8}
        

        # self.possible_actions_map = {
        #     1: [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]],
        #     2: [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]],
        #     3: [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0]],
        #     4: [[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]],
        #     5: [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1]],
        #     6: [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]],
        #     7: [[-1, 0], [-1, -1], [0, -1], [-1, -1], [-1, 0]],
        #     8: [[1, -1], [0, -1], [-1, -1], [-1, 0], [-1, -1]]
        #
        # }

        self.possible_actions_map = {
            1: [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]],
            2: [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]],
            3: [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0]],
            4: [[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]],
            5: [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1]],
            6: [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]],
            7: [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]],
            8: [[1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]

        }

        self.actionvalue_heading_action = {
            0: {1: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                2: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                3: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                4: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                5: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                6: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                7: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                8: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)'},
            1: {1: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                2: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                3: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                4: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                5: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                6: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                7: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                8: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)'},
            2: {1: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                2: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                3: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                4: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                5: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                6: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                7: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                8: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)'},
            3: {1: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                2: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                3: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                4: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                5: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                6: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                7: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                8: 'self.take_action(delta_alt=-1,delta_x=-0,delta_y=-1,new_heading=1)'},
            4: {1: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                2: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                3: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                4: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                5: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                6: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                7: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                8: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)'},
            5: {1: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)',
                2: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=-1, new_heading=8)',
                3: 'self.take_action(delta_alt=0, delta_x=0, delta_y=-1, new_heading=1)',
                4: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=2)',
                5: 'self.take_action(delta_alt=0, delta_x=1, delta_y=0, new_heading=3)',
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
                 5: 'self.take_action(delta_alt=1, delta_x=1, delta_y=0, new_heading=3)',
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
                 8: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=2)'},
            15: {1: 'self.drop_package()',
                 2: 'self.drop_package()',
                 3: 'self.drop_package()',
                 4: 'self.drop_package()',
                 5: 'self.drop_package()',
                 6: 'self.drop_package()',
                 7: 'self.drop_package()',
                 8: 'self.drop_package()', }

        }

        print("here")

    def neighbors(self, arr, x, y, N):

        # https://stackoverflow.com/questions/32604856/slicing-outside-numpy-array
        # new_arr = np.zeros((N,N))

        left_offset = x - N // 2
        top_offset = y - N // 2

        # These are the 4 corners in real world coords
        left = max(0, x - N // 2)
        right = min(arr.shape[0], x + N // 2)
        top = max(0, y - N // 2)
        bottom = min(arr.shape[1], y + N // 2)

        window = arr[left:right + 1, top:bottom + 1]

        # newArr = np.zeros(self.original_map_volume['vol'][0].shape)
        # newArr[x-N//2:x+N//2+1,y-N//2:y+N//2+1] = window
        # return newArr
        return [window, left, top, right, bottom]

    def position_value(self, terrain, altitude, reward_dict, probability_dict):
        damage_probability = probability_dict['damage_probability'][altitude]
        # if terrain in probability_dict['stuck_probability'].keys():
        #     stuck_probability = probability_dict['stuck_probability'][terrain]
        # else:
        #     stuck_probability = 0.0
        # if terrain in probability_dict['sunk_probability'].keys():
        #     sunk_probability = probability_dict['sunk_probability'][terrain]
        # else:
        #     sunk_probability = 0.0
        damaged = np.random.random() < damage_probability
        # stuck = np.random.random() < stuck_probability
        # sunk = np.random.random() < sunk_probability
        self.package_state = 'DAMAGED' if damaged else 'OK'
        # package_state += '_STUCK' if stuck else ''
        # package_state += '_SUNK' if sunk else ''
        print("Package state:", self.package_state)
        reward = reward_dict[self.package_state]
        return reward

    def drop_package(self):
        self.drop = True
        alt = self.altitude
        drone_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        region = self.drop_package_grid_size_by_alt[self.altitude]
        neighbors, left, top, right, bottom = self.neighbors(self.original_map_volume['vol'][0], int(drone_position[1]),
                                              int(drone_position[2]), region)
        w = self.map_volume['vol'][0][left:right, top:bottom]
        is_hiker_in_neighbors = np.any(w == self.map_volume['feature_value_map']['hiker']['val'])
        # print("neigh:")
        # print(neighbors)
        # x = drone_position[0]
        # y = drone_position[1]
        x = np.random.randint(0, neighbors.shape[0])
        y = np.random.randint(0, neighbors.shape[1])
        #print(x, y)
        # value = [x, y]
        value = neighbors[x, y] # It returns what kind of terrain is there in (number)
        pack_world_coords = (x + left, y + top) #(x,y) #
        terrain = self.original_map_volume['value_feature_map'][value]['feature'] # what kind of terrain is there (string)
        reward = self.position_value(terrain, alt, self.drop_rewards, self.drop_probabilities)
        self.pack_dist = np.linalg.norm(np.array(pack_world_coords) - np.array(self.hiker_position[-2:])) # Check it out if it is correct!!!
        reward -= self.pack_dist * 0.1 # Penalizing according to the dropping distance is wrong! The closer the package is the HIGHER the penalty
        #reward = reward*(1/(self.pack_dist+1e-7))*0.1
        self.reward = reward*is_hiker_in_neighbors # YOU CANNOT DO THAT EVEN IF IT WORKS FOR THAT MAP AS IT DOESNT GET PENALTY FOR DAMAGING THE PACK!
        #print(terrain, reward)

    def take_action(self, delta_alt=0, delta_x=0, delta_y=0, new_heading=1):
        # print("stop")
        vol_shape = self.map_volume['vol'].shape

        local_coordinates = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        if int(local_coordinates[1]) + delta_y < 0 or \
                int(local_coordinates[2]) + delta_x < 0 or \
                int(local_coordinates[1] + delta_y > vol_shape[1] - 1) or \
                int(local_coordinates[2] + delta_x > vol_shape[2] - 1):
            return 0

        # todo update with shape below
        forbidden = [(0, 0), (vol_shape[1] - 1, 0),
                     (vol_shape[1] - 1, vol_shape[1] - 1), (0, vol_shape[1] - 1)]
        #print((int(local_coordinates[1]) + delta_y, int(local_coordinates[2]) + delta_x), forbidden)
        if (int(local_coordinates[1]) + delta_y, int(local_coordinates[2]) + delta_x) in forbidden:
            return 0

        new_alt = self.altitude + delta_alt if self.altitude + delta_alt < 4 else 3
        if new_alt < 0:
            return 0

        # put back the original
        self.map_volume['vol'][self.altitude][local_coordinates[1], local_coordinates[2]] = float(
            self.original_map_volume['vol'][local_coordinates])

        # self.map_volume['flat'][local_coordinates[1],local_coordinates[2]] = float(self.original_map_volume['flat'][local_coordinates[1],local_coordinates[2]])
        # self.map_volume['img'][local_coordinates[1],local_coordinates[2]] = self.original_map_volume['img'][local_coordinates[1],local_coordinates[2]]
        # put the hiker back
        self.map_volume['vol'][self.hiker_position] = self.map_volume['feature_value_map']['hiker']['val']
        # self.map_volume['flat'][self.hiker_position[1],self.hiker_position[2]] = self.map_volume['feature_value_map']['hiker']['val']
        # self.map_volume['img'][self.hiker_position[1],self.hiker_position[2]] = self.map_volume['feature_value_map']['hiker']['color']
        # put the drone in
        # self.map_volume['flat'][local_coordinates[1]+delta_y,local_coordinates[2]+delta_x] = self.map_volume['feature_value_map']['drone'][new_alt]['val']
        self.map_volume['vol'][new_alt][local_coordinates[1] + delta_y, local_coordinates[2] + delta_x] = \
        self.map_volume['feature_value_map']['drone'][new_alt]['val']
        # self.map_volume['img'][local_coordinates[1] + delta_y, local_coordinates[2] + delta_x] = self.map_volume['feature_value_map']['drone'][new_alt]['color']
        # for i in range(4,-1,-1):
        #     if self.map_volume['vol'][i][local_coordinates[1],local_coordinates[2]]:
        #         self.map_volume['flat'][int(local_coordinates[1]),int(local_coordinates[2])] = float(self.map_volume['vol'][i][int(local_coordinates[1]),int(local_coordinates[2])])
        #         break
        self.altitude = new_alt
        self.heading = new_heading

        if self.real_actions:
            drone_position = np.where(
                self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])

            success = self.mavsimhandler.fly_path(coordinates=[self.reference_coordinates[0] + int(drone_position[1]),
                                                               self.reference_coordinates[1] + int(drone_position[2])],
                                                  altitude=self.altitude)

        return 1

    def check_for_hiker(self):
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        # hiker_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['hiker'][0])
        # print("drone",drone_position)
        # print("hiker",self.hiker_position)
        # drone or hiker coords format (alt,x,y)
        if (drone_position[1], drone_position[2]) == (self.hiker_position[1], self.hiker_position[2]):
            return 1
        return 0
        # return int(self.map_volume[0]['hiker'][int(local_coordinates[0]),int(local_coordinates[1])])

    def check_for_crash(self):
        # if drone on altitude 0, crash
        if self.altitude == 0:
            return 1

        # if len(self.map_volume[0]['drone'].nonzero()[0]):
        #     return 1
        # at any other altutidue, check for an object at the drone's position
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        return int(self.original_map_volume['vol'][drone_position])
        # drone_position = self.map_volume[self.altitude]['drone'].nonzero()
        # for i in range(self.altitude,4):
        #
        #     for key in self.map_volume[i]:
        #         if key == 'drone' or key == 'map':
        #             continue
        #         #just check if drone position is returns a non-zero
        #         if self.map_volume[i][key][int(drone_position[0]),int(drone_position[1])]:
        #             return 1
        # return 0

    #PREVIOUS WORKING STEP

    # def step(self, action):
    #         ''' return next observation, reward, finished, success '''
    #
    #         action = int(action)
    #         info = {}
    #         info['success'] = False
    #
    #         done = False
    #         drone = np.where(
    #             self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
    #         hiker = self.hiker_position
    #         # You should never reach as this is the state(t-1) distane. After eval you get the new distance
    #         self.dist = np.linalg.norm(np.array(drone[-2:]) - np.array(hiker[-2:]))  # we remove height from the equation so we avoid going diagonally down
    #
    #         # Here the action takes place
    #         x = eval(self.actionvalue_heading_action[action][self.heading])
    #         # A new observation is generated which we do not see cauz we reset() and render in the step function
    #         observation = self.generate_observation()
    #
    #         crash = self.check_for_crash()
    #         info['success'] = not crash
    #         #self.render()
    #
    #         if crash:
    #             reward = -1
    #             done = True
    #             print("CRASH")
    #             if self.restart_once_done:  # HAVE IT ALWAYS TRUE!!! It learned the first time WITHOUT RESETING FROM CRASH
    #                 #observation = self.reset()
    #                 return (observation, reward, done, info)
    #             # return (self.generate_observation(), reward, done, info)
    #         # if self.dist < self.dist_old:
    #         #     reward = 1 / self.dist  # Put it here to avoid dividing by zero when you crash on the hiker
    #         # else:
    #         #     reward = -1 / self.dist
    #         if self.check_for_hiker():
    #             done = True
    #             reward = 1# + self.alt_rewards[self.altitude]
    #             # reward = 1 + 1 / self.dist
    #             print('SUCCESS!!!')
    #             if self.restart_once_done:  # HAVE IT ALWAYS TRUE!!!
    #                 #observation = self.reset()
    #                 return (observation, reward, done, info)
    #         # print("state", [ self.observation[self.altitude]['drone'].nonzero()[0][0],self.observation[self.altitude]['drone'].nonzero()[1][0]] )
    #         self.dist_old = self.dist
    #         #reward = (self.alt_rewards[self.altitude] * 0.1) * ( 1/((self.dist** 2) + 1e-7) )  # -0.01 + # previous reward = (self.alt_rewards[self.altitude] * 0.1) * ( 1 / self.dist** 2 + 1e-7 )  # -0.01 + #
    #         reward = -0.01 # If you put -0.1 then it prefers to go down and crash all the time for (n-step=32)!!!
    #         return (observation, reward, done, info)

    def step(self, action):
        ''' return next observation, reward, finished, success '''

        action = int(action)
        info = {}
        info['success'] = False

        done = False
        drone_old = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        hiker = self.hiker_position
        # Do the action (drone is moving)
        x = eval(self.actionvalue_heading_action[action][self.heading])

        observation = self.generate_observation()
        drone = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        self.dist = np.linalg.norm(np.array(drone[-2:]) - np.array(hiker[-2:])) # we remove height from the equation so we avoid going diagonally down

        crash = self.check_for_crash()
        info['success'] = not crash

        # BELOW WAS WORKING FINE FOR FINDING HIKER
        # reward = (self.alt_rewards[self.altitude]*0.1)*(1/self.dist**2+1e-7)# + self.drop*self.reward (and comment out the reward when you drop and terminate episode
        #reward = (self.alt_rewards[self.altitude]*0.1)*((1/(self.dist**2)+1e-7)) # -0.01 + # The closer we are to the hiker the more important is to be close to its altitude
        if crash:
            reward = -1
            done = True
            print("CRASH")
            if self.restart_once_done: # HAVE IT ALWAYS TRUE!!! It learned the first time WITHOUT RESETING FROM CRASH
                return (observation, reward, done, info)
            #return (self.generate_observation(), reward, done, info)
        # if self.dist < self.dist_old:
        #     reward = 1 / self.dist  # Put it here to avoid dividing by zero when you crash on the hiker
        # else:
        #     reward = -1 / self.dist
        if self.drop:#self.check_for_hiker():
            done = True
            #reward = 1 + self.alt_rewards[self.altitude] # THIS WORKS FOR FINDING THE HIKER
            if self.check_for_hiker():
                reward = 1 + self.reward + self.alt_rewards[self.altitude]
            else:
                reward = self.reward + self.alt_rewards[self.altitude] # (try to multiply them and see if it makes a difference!!! Here tho u reward for dropping low alt
            print('DROP!!!', 'self.reward=', self.reward, 'alt_reward=', self.alt_rewards[self.altitude])
            if self.restart_once_done: # HAVE IT ALWAYS TRUE!!!
                return (observation, reward, done, info)
        # print("state", [ self.observation[self.altitude]['drone'].nonzero()[0][0],self.observation[self.altitude]['drone'].nonzero()[1][0]] )
        self.dist_old = self.dist
        # HERE YOU SHOULD HAVE THE REWARD IN CASE IT CRASHES AT ALT=0 OR IN GENERAL AFTER ALL CASES HAVE BEEN CHECKED!!!
        if self.check_for_hiker(): # On top of the hiker
            #print("hiker found:", self.check_for_hiker())
            # reward = (self.alt_rewards[self.altitude]*0.1)*(1/self.dist**2+1e-7) + self.drop*self.reward (and comment out the reward when you drop and terminate episode
            reward = 0 #1 + self.alt_rewards[self.altitude]
        else:
            # We don't want the drone to wonder around away from the hiker so we keep it close
            # The reward below though with PPO will make the drone just going close and around the hiker forever as it gather reward all the time
            reward = (self.alt_rewards[self.altitude]*0.1)*((1/((self.dist**2)+1e-7))) # -0.01 + # The closer we are to the hiker the more important is to be close to its altitude
            #print("scale:",(1/((self.dist**2+1e-7))), "dist=",self.dist+1e-7, "alt=", self.altitude, "drone:",drone, "hiker:", hiker,"found:", self.check_for_hiker())
        return (self.generate_observation(), reward, done, info)

    def reset(self):
        self.dist_old = 1000
        self.drop = False
        self.heading = random.randint(1, 8)
        self.altitude = 2
        self.reward = 0
        _map = random.choice(self.maps)
        self.map_volume = CNP.map_to_volume_dict(_map[0], _map[1], 10, 10)
        # Set hiker's and drone's locations
        #hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 1), random.randint(2, self.map_volume['vol'].shape[1] - 2)) #(8,8) #
        #if self.dropping:
        hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 2), random.randint(2, self.map_volume['vol'].shape[1] - 2))  # (7,8) #
        drone = random.choice([(hiker[0]-1, hiker[1]-1),(hiker[0]-1, hiker[1]),(hiker[0], hiker[1]-1)])## Package drop starts close to hiker!!! #(random.randint(2, self.map_volume['vol'].shape[1] - 1), random.randint(2, self.map_volume['vol'].shape[1] - 2)) # (8,8) #
        #else:
            # hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 2), random.randint(2, self.map_volume['vol'].shape[1] - 2))  # (7,8) #
            # drone = (random.randint(2, self.map_volume['vol'].shape[1] - 2), random.randint(2, self.map_volume['vol'].shape[1] - 2))

        while drone == hiker:
            print('$$$$$$$$ AWAY !!! $$$$$$$')
            drone = (random.randint(2, self.map_volume['vol'].shape[1] - 1),
                     random.randint(2, self.map_volume['vol'].shape[1] - 2))

        self.original_map_volume = copy.deepcopy(self.map_volume)

        # self.local_coordinates = [local_x,local_y]
        # self.world_coordinates = [70,50]
        self.reference_coordinates = [_map[0], _map[1]]


        self.real_actions = False
        # put the drone in
        self.map_volume['vol'][self.altitude][drone[0], drone[1]] = \
        self.map_volume['feature_value_map']['drone'][self.altitude]['val']
        self.map_volume['flat'][drone[0], drone[1]] = self.map_volume['feature_value_map']['drone'][self.altitude][
            'val']
        self.map_volume['img'][drone[0], drone[1]] = self.map_volume['feature_value_map']['drone'][self.altitude][
            'color']
        # self.map_volume[altitude]['drone'][local_y, local_x] = 1.0
        # put the hiker in@ altitude 0
        self.map_volume['vol'][0][hiker[0], hiker[1]] = self.map_volume['feature_value_map']['hiker']['val']
        self.map_volume['flat'][hiker[0], hiker[1]] = self.map_volume['feature_value_map']['hiker']['val']
        self.map_volume['img'][hiker[0], hiker[1]] = self.map_volume['feature_value_map']['hiker']['color']
        self.hiker_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['hiker']['val'])

        self.image_layers[0] = self.create_image_from_volume(0)
        self.image_layers[1] = self.create_image_from_volume(1)
        self.image_layers[2] = self.create_image_from_volume(2)
        self.image_layers[3] = self.create_image_from_volume(3)
        self.image_layers[4] = self.create_image_from_volume(4)

        observation = self.generate_observation()
        self.render()
        return observation

    def plane_image(self, heading, color):
        '''Returns a 5x5 image as np array'''
        for point in self.planes[heading][0]:
            self.planes[heading][1][point[0], point[1]] = color
        return self.planes[heading][1]

    def create_image_from_volume(self, altitude):
        canvas = np.zeros((self.map_volume['vol'].shape[1], self.map_volume['vol'].shape[1], 3), dtype=np.uint8)
        og_vol = self.original_map_volume
        combinations = list(itertools.product(range(0, canvas.shape[0]), range(0, canvas.shape[0])))
        for x, y in combinations:
            if og_vol['vol'][altitude][x, y] == 0.0:
                canvas[x, y, :] = [255, 255, 255]
            else:
                canvas[x, y, :] = og_vol['value_feature_map'][og_vol['vol'][altitude][x, y]]['color']

        return imresize(canvas, self.factor * 100, interp='nearest')

    def create_nextstep_image(self):
        canvas = np.zeros((5, 5, 3), dtype=np.uint8)
        slice = np.zeros((5, 5))
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        drone_position_flat = [int(drone_position[1]), int(drone_position[2])]
        # hiker_found = False
        # hiker_point = [0, 0]
        # hiker_background_color = None
        column_number = 0
        for xy in self.possible_actions_map[self.heading]:
            try:
                # no hiker if using original
                column = self.map_volume['vol'][:, drone_position_flat[0] + xy[0], drone_position_flat[1] + xy[1]]
                # for p in column:
                #     #print(p)
                #     #print(p == 50.0)
                #     if p == 50.0: # Hiker representation in the volume
                #         #print("setting hiker_found to True")
                #         hiker_found = True
                #
                # if hiker_found:
                #     val = self.original_map_volume['vol'][0][
                #         drone_position_flat[0] + xy[0], drone_position_flat[1] + xy[1]]
                #     hiker_background_color = self.original_map_volume['value_feature_map'][val]['color']
                #     # column = self.original_map_volume['vol'][:,drone_position_flat[0]+xy[0],drone_position_flat[1]+xy[1]]
            except IndexError:
                column = [1., 1., 1., 1., 1.]
            slice[:, column_number] = column
            column_number += 1
            #print("ok")
        # put the drone in
        # cheat
        slice[self.altitude, 2] = int(self.map_volume['vol'][drone_position])
        combinations = list(itertools.product(range(0, canvas.shape[0]), range(0, canvas.shape[0])))
        for x, y in combinations:
            if slice[x, y] == 0.0:
                canvas[x, y, :] = [255, 255, 255]
            # elif slice[x, y] == 50.0:
            #     canvas[x, y, :] = hiker_background_color
            #     hiker_point = [x, y]
            else:
                canvas[x, y, :] = self.map_volume['value_feature_map'][slice[x, y]]['color']

        # increase the image size, then put the hiker in
        canvas = imresize(canvas, self.factor * 100, interp='nearest')
        # hiker_position = (int(self.hiker_position[1] * 5), int(self.hiker_position[2]) * 5)
        # map[hiker_position[0]:hiker_position[0]+5,hiker_position[1]:hiker_position[1]+5,:] = self.hiker_image
        # print("hiker found", hiker_found)
        # print("hiker_point", hiker_point)
        # if hiker_found:
        #     for point in self.hikers[0][0]:
        #         canvas[hiker_point[0] * self.factor + point[0], hiker_point[1] * self.factor + point[1], :] = \
        #             self.map_volume['feature_value_map']['hiker']['color']

        return imresize(np.flip(canvas, 0), 200, interp='nearest')

    def generate_observation(self):
        obs = {}
        obs['volume'] = self.map_volume
        image_layers = copy.deepcopy(self.image_layers)
        map = copy.deepcopy(self.original_map_volume['img'])

        # put the drone in the image layer
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        drone_position = (int(drone_position[1]) * self.factor, int(drone_position[2]) * self.factor)
        for point in self.planes[self.heading][0]:
            image_layers[self.altitude][drone_position[0] + point[0], drone_position[1] + point[1], :] = \
            self.map_volume['feature_value_map']['drone'][self.altitude]['color']

        # put the hiker in the image layers
        hiker_position = (int(self.hiker_position[1] * self.factor), int(self.hiker_position[2]) * self.factor)
        for point in self.hikers[0][0]:
            image_layers[0][hiker_position[0] + point[0], hiker_position[1] + point[1], :] = \
            self.map_volume['feature_value_map']['hiker']['color']

        # map = self.original_map_volume['img']
        map = imresize(map, self.factor * 100, interp='nearest')  # resize by factor of 5
        # add the hiker
        hiker_position = (int(self.hiker_position[1] * 5), int(self.hiker_position[2]) * 5)
        # map[hiker_position[0]:hiker_position[0]+5,hiker_position[1]:hiker_position[1]+5,:] = self.hiker_image
        for point in self.hikers[0][0]:
            map[hiker_position[0] + point[0], hiker_position[1] + point[1], :] = \
            self.map_volume['feature_value_map']['hiker']['color']
        # add the drone
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        drone_position = (int(drone_position[1]) * 5, int(drone_position[2]) * 5)
        for point in self.planes[self.heading][0]:
            map[drone_position[0] + point[0], drone_position[1] + point[1], :] = \
            self.map_volume['feature_value_map']['drone'][self.altitude]['color']
        # map[drone_position[0]:drone_position[0] + 5,drone_position[1]:drone_position[1] + 5] = self.plane_image(self.heading,self.map_volume['feature_value_map']['drone'][self.altitude]['color'])

        # map = imresize(map, (1000,1000), interp='nearest')

        '''vertical slices at drone's position'''
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        # slice1 = np.flip(self.map_volume['vol'][:, int(drone_position[1]), :], 0)
        # # slice1 = np.flip(slice1,1)
        # slice2 = np.flip(self.map_volume['vol'][:, :, int(drone_position[2])], 0)
        # # slice2 = np.flip(slice2, 1)
        # obs['slices'] = [slice1, slice2]
        #
        # # slices as images
        # slice1_img = np.zeros((5, slice1.shape[1], 3),
        #                       dtype=np.uint8)  # canvas = np.zeros((self.map_volume['vol'].shape[1], self.map_volume['vol'].shape[1], 3), dtype=np.uint8)
        # combinations = list(itertools.product(range(0, 5), range(0, slice1_img.shape[1])))
        # for x, y in combinations:
        #     a = slice1[x, y]
        #     if slice1[x, y] == 0.0:
        #         slice1_img[x, y, :] = [255, 255, 255]
        #     else:
        #         slice1_img[x, y, :] = self.original_map_volume['value_feature_map'][slice1[x, y]]['color']
        #
        # slice2_img = np.zeros((5, slice2.shape[1], 3), dtype=np.uint8)
        # combinations = list(itertools.product(range(0, 5), range(0, slice2_img.shape[1])))
        # for x, y in combinations:
        #     if slice2[x, y] == 0.0:
        #         slice2_img[x, y, :] = [255, 255, 255]
        #     else:
        #         slice2_img[x, y, :] = self.original_map_volume['value_feature_map'][slice2[x, y]]['color']
        # obs['slice_images'] = [slice1_img, slice2_img]

        nextstepimage = self.create_nextstep_image()
        obs['nextstepimage'] = nextstepimage
        obs['img'] = map
        obs['image_layers'] = image_layers
        return obs

    def render(self, mode='human', close=False):

        # return
        if self.verbose == False:
           return
        # img = self.observation
        # map = self.original_map_volume['img']
        obs = self.generate_observation()
        self.map_image = obs['img']
        self.alt_view = obs['nextstepimage']
        # fig = plt.figure(self.this_fig_num)
        # img = np.zeros((20,20,3))
        # img[10,10,0] = 200
        # img[10,10,1] = 153
        # img[10,10,2] = 255

        #fig = plt.figure(0)
        #fig1 = plt.figure(1)
        #plt.clf()
        # plt.subplot(211)
        # plt.imshow(self.map_image)
        # plt.subplot(212)
        # plt.imshow(self.alt_view)
        # #fig.canvas.draw()
        # #plt.show()
        # plt.pause(0.00001)
        return

    def _close_env(self):
        plt.close(1)
        return


# a = GridworldEnv(map_x=70, map_y=50, local_x=2, local_y=2, hiker_x=10, heading=1, altitude=3)
# a.reset()
# # a.step(12)
# #
# # def show_img():
#
# for i in range(10000):
#     a.step(random.randint(1, 14))
#     # local_coordinates = a.map_volume[a.altitude]['drone'].nonzero()
#     # print("coordinates", local_coordinates, a.heading)
#     if a.check_for_crash():
#         print("crash at altitude", a.altitude)
#         a.reset()
#         time.sleep(0.5)
#     if a.check_for_hiker():
#         print("hiker after", i)
#         break

# print(a.check_for_crash())
print('complete')