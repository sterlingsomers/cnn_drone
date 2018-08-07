import os
import pickle
import numpy as np
from gym_gridworld.envs.mapquery import terrain_request
from pathlib import Path
path='./gym_gridworld/'
# map = pickle.load(open('050070.dict','rb'))
# top_left = (50,70)
# #make the volume
# vol = np.zeros((5,20,20))
# feature_value_map = {} #{[alt,feature]:value}
# value_feature_map = {} #{value:(alt,feature)}
# index = 1
# #alt_index = {0:1,1:1,2:1,3:1,4:1}
# for xy, feat in map.items():
#     if feat not in list(feature_value_map.keys()):
#         feature_value_map[feat] = index#alt_index[feat[0]]
#         value_feature_map[index] = feat
#         index+=1#alt_index[feat[0]] += 1
#
#     vol[feat[0],xy[0]-top_left[0],xy[1]-top_left[1]] = feature_value_map[feat]
#
#

def get_feature_value_maps(x,y,map):
    '''This will create, load, and expand a feature-> value dictionary and
    a value-> feature dictionary.
    Will return 2 dictionaries.'''
    feature_value_map = {}
    value_feature_map = {}
    #first check for existing feature maps
    feature_to_value = Path('features/features_to_values.dict')
    value_to_feature = Path('features/values_to_features.dict')

    if feature_to_value.is_file():
        feature_value_map = pickle.load(open(feature_to_value,'rb'))
    if value_to_feature.is_file():
        value_feature_map = pickle.load(open(value_to_feature, 'rb'))


    return (feature_value_map, value_feature_map)

def convert_map_to_volume_dict(x,y,map,width,height):
    return_dict = {}
    top_left = (x,y)
    feature_value_map = {}
    img = np.zeros((width,height,3),dtype=np.uint8)
    vol = np.zeros((5, width, height))
    flat = np.zeros((width,height))
    # color_map = {1:[153,255,153],2:[156,73,0],3:[0,204,0],
    #              4:[0,102,51],5:[135,135,0],6:[202,202,0],
    #              7:[255,255,0], 8:[255,180,0], 9:[200,5,0],50:[255,0,0]}
    color_map = {'pine tree':[0,100,14],'pine trees':[0,172,23],'grass':[121,151,0],
                 'bush':[95,98,57],'bushes':[164,203,8],'trail':[145,116,0],
                 'water':[0,34,255],
                 'drone':{0:[102,0,51],1:[153,0,153],2:[255,51,255],3:[255,153,255],4:[255,0,0]},
                 'hiker':[255,0,0]}
    #load value maps: feature -> value and value -> feature
    #feature_value_map = {} #{[alt,feature]:value}
    #value_feature_map = {} #{value:(alt,feature)}
    feature_value_map,value_feature_map = get_feature_value_maps(x,y,map)
    if list(value_feature_map.keys()):
        value = max(list(value_feature_map.keys())) + 1
    else:
        value = 1.0
    for xy, feat in map.items():
        if feat[1] not in list(feature_value_map.keys()):
            #feature_value_map[feat[1]] = {}

            #for i in range(5):
            feature_value_map[feat[1]] = {'val': value, 'color':color_map[feat[1]]}
            value_feature_map[value] = {'feature':feat[1], 'alt':float(feat[0]), 'color':color_map[feat[1]]}
            value += 1

            #value += 20
        #put it in the flat
        flat[xy[1] - top_left[1], xy[0] - top_left[0]] = feature_value_map[feat[1]]['val']
        img[xy[1]- top_left[1], xy[0] - top_left[0], :] = feature_value_map[feat[1]]['color']
        #project it downwards through the volume
        for z in range(feat[0],-1,-1):
            vol[z,xy[1] - top_left[1],xy[0] - top_left[0]] = feature_value_map[feat[1]]['val']



    return_dict['feature_value_map'] = feature_value_map
    return_dict['value_feature_map'] = value_feature_map
    #save before returning
    #todo fix value_feature_map and feature_maps -> they should be the same (except inside out)
    # print("saving value/feature maps")
    # with open(path+'features/features_to_values.dict', 'wb') as handle:
    #     pickle.dump(feature_value_map, handle)
    # with open(path+'features/values_to_features.dict', 'wb') as handle:
    #     pickle.dump(value_feature_map,handle)

    return_dict['vol'] = vol
    return_dict['flat'] = flat
    return_dict['img'] = img

    #add the hiker and the drone
    feature_value_map['hiker'] = {}
    feature_value_map['drone'] = {}
    #drone
    # value += 20
    value = max(list(value_feature_map.keys())) + 1
    for i in range(5):
        feature_value_map['drone'][i] = {'val': value, 'color': color_map['drone'][i]}
        value_feature_map[value] = {'feature': 'drone', 'alt':i, 'color':color_map['drone'][i]}
        value += 1



    #hiker - reserving 50
    value = 50#max(list(value_feature_map.keys())) + 20

    #for i in range(5):
    feature_value_map['hiker']['val'] = value
    feature_value_map['hiker']['color'] = color_map['hiker']
    value_feature_map[value] = {'feature':'hiker', 'alt':0, 'color':color_map['hiker']}






    # for i in range(len(vol)):
    #     key_string = i#'alt{}'.format(i)
    #     if key_string not in return_dict:
    #         return_dict[key_string] = {}
    #         return_dict[key_string]['map'] = vol[i]
    #
    #     #what features are at that altitude?
    #     current_features = []
    #     for value,feature in value_feature_map.items():
    #         if feature[0] == i:
    #             current_features.append(feature[1])
    #
    #     current_features.append('drone')
    #     if i == 0:
    #         current_features.append('hiker')
    #
    #     for current_feature in current_features:
    #         return_dict[key_string][current_feature] = np.zeros((20,20))
    #
    #     non_zeros = np.transpose(vol[i].nonzero())
    #     for non_zero in non_zeros:
    #         feature = value_feature_map[vol[i][non_zero[0]][non_zero[1]]][1]
    #         #return_dict[key_string][feature][non_zero[0][non_zero[1]]] = 1.0
    #         return_dict[key_string][feature][non_zero[0],non_zero[1]] = 1.0
    #         #buil

    return return_dict


def map_to_volume_dict(x=0,y=0,width=5,height=5):
    #does the map already exist in the maps/ folder?
    return_dict = {}
    filename = '{}-{}.mp'.format(x,y)
    maps = []
    map = 0
    for files in os.listdir(path+'maps'):
        if files.endswith(".mp"):
            maps.append(files)
    #loops through because I'll need the actual map
    for files in maps:
        if filename == files:
            #print("loading existing map.")
            map = pickle.load(open(path+'maps/' + filename,'rb'))
    if not map:
        print("generating map. YOU NEED MAVSIM RUNNING!!!")
        map = terrain_request(x,y,width,height)

        #store it for future use
        print("saving map.")
        with open(path+'maps/' + filename, 'wb') as handle:
            pickle.dump(map, handle)
    #convert_map_to_volume_dict(x,y,map)
    return convert_map_to_volume_dict(x,y,map,width,height)

 #   return return_dict



#sample code
#a = map_to_volume_dict(90,70,10,10)
# f,v = get_feature_value_maps(300,200,a) #300,200
# print('complete.')