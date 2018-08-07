import json
import pickle
import pdb

import numpy as np

from common import render_map


def load_terrain_file(file_path):
    with open(file_path, 'r') as load_file:
        terrain_dict = json.load(load_file)
    parsed_terrain_dict = {}
    for key, value in terrain_dict.items():
        key = tuple([int(coord) for coord in key.split('_')])
        parsed_terrain_dict[key] = tuple(value)
    return parsed_terrain_dict


def terrain_dict_to_gridworld_map(terrain_dict):
    map_coords = list(terrain_dict.keys())
    longitudes = [coord[0] for coord in map_coords]
    latitudes = [coord[1] for coord in map_coords]
    lower_longitude = min(longitudes)
    upper_longitude = max(longitudes)
    lower_latitude = min(latitudes)
    upper_latitude = max(latitudes)
    longitude_dimension = upper_longitude - lower_longitude + 1
    latitude_dimension = upper_latitude - lower_latitude + 1
    altitude_dimension = 4
    gridworld_map = np.full(shape=(longitude_dimension, latitude_dimension, altitude_dimension), fill_value=-1)
    gridworld_dict = populate_gridworld(gridworld_map, terrain_dict, lower_longitude, lower_latitude)
    return gridworld_dict


def populate_gridworld(gridworld_map, terrain_dict, lower_lon, lower_lat):
    terrain_types = find_terrain_types(terrain_dict)
    for key, value in terrain_dict.items():
        lon, lat = key
        lon -= lower_lon
        lat -= lower_lat
        max_altitude, terrain_type = value
        if max_altitude < 0:
            print("lon: {}, lat: {}, max_altitude: {}, terrain_type: {}".format(lon, lat, max_altitude, terrain_type))
            max_altitude = 0
        for altitude in range(0, max_altitude + 1):
            if altitude > 3:
                altitude = 3
                # print("terrain type:", terrain_type)
            gridworld_map[lon, lat, altitude] = terrain_types.index(terrain_type)
    gridworld_dict = {'map': gridworld_map, 'types': terrain_types}
    return gridworld_dict


def find_terrain_types(terrain_dict):
    terrain_types = []
    for key, value in terrain_dict.items():
        if value[1] not in terrain_types:
            terrain_types.append(value[1])
    return terrain_types


def save_gridworld(gridworld_dict, save_file_path):
    with open(save_file_path, 'wb') as save_file:
        pickle.dump(gridworld_dict, save_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_json_file(load_file_path):
    with open(load_file_path, 'r') as load_file:
        json_dict = json.load(load_file)
    return json_dict


def main():
    load_file_path = '../data/world_map_terrain.json'
    save_file_path = 'gridworld.pkl'
    terrain_dict = load_terrain_file(load_file_path)
    gridworld_dict = terrain_dict_to_gridworld_map(terrain_dict)
    game_of_drones_constants_file_path = '../data/game_of_drones_constants.json'
    game_of_drones_rewards_file_path = '../data/game_of_drones_rewards.json'
    gridworld_dict['drop_probabilities'] = load_json_file(game_of_drones_constants_file_path)
    gridworld_dict['drop_rewards'] = load_json_file(game_of_drones_rewards_file_path)
    image_load_path = '../data/world-4096.png'
    tiles_array = render_map.create_tiles(image_load_path, (500, 500))
    tiles_array = render_map.flipud_tiles_array(tiles_array)
    tiles_array = render_map.rotate_tiles_array(tiles_array)
    gridworld_dict['tiles'] = tiles_array
    pdb.set_trace()
    save_gridworld(gridworld_dict, save_file_path)


if __name__ == '__main__':
    main()