import pickle
import numpy as np

all_data = ''
with open('all_data.pkl', 'rb') as handle:
    all_data = pickle.load(handle)
#each entry in all_data is a mission
#each entry in a mission is a step
#the dictionary is the data from that step



def get_drone_position(map_volume,drone_altitude):
    """This function will return coordinates of the drone in the volume.
    map_volume is the entire volume dictionary.
    drone altitude is an integer 0-3, used to index the value in the volume."""
    drone_position = np.where(map_volume['vol'] == map_volume['feature_value_map']['drone'][drone_altitude]['val'])
    return drone_position

def process_step(step_data):
    #step_data is a dictionary
    #returns a dictionary describing the step
    return_dict = {"hiker_position":None,
                   "drone_position":None,
                   "drone_altitude":None,
                   "drone_heading":None,
                   "hiker_heading":None,
                   "fc":None}

    return_dict['drone_altitude'] = step_data['drone_altitude']
    return_dict['drone_heading'] = step_data['drone_heading']
    return_dict['fc'] = step_data['fc']
    return_dict['drone_position'] = get_drone_position(step_data['map_volume'],step_data['drone_altitude'])

    return return_dict


#cycle through the missions, then each step in the mission

print("done.")