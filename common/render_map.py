import pdb
import os
from copy import deepcopy
from PIL import Image
import math

import numpy as np
from scipy import ndimage


def load_image_array(image_file_path):
    map_arr = ndimage.imread(image_file_path)
    return map_arr


def create_tiles(image_file_path, gridworld_dimensions, remove_alpha=True):
    map_arr = load_image_array(image_file_path)
    if remove_alpha:
        map_arr = map_arr[:, :, :3]  # Remove alpha channel.
    map_arr = np.rot90(np.rot90(map_arr))
    tile_delta = (map_arr.shape[0] / gridworld_dimensions[0], map_arr.shape[1] / gridworld_dimensions[1])
    tile_shape = (math.floor(tile_delta[0]), math.floor(tile_delta[1]))
    tile_arrays = []
    for x_value in range(gridworld_dimensions[0]):
        tile_array_column = []
        tile_x_lower = round(x_value * tile_delta[0])
        for y_value in range(gridworld_dimensions[1]):
            tile_y_lower = round(y_value * tile_delta[1])
            tile_x_upper = tile_x_lower + tile_shape[0]
            tile_y_upper = tile_y_lower + tile_shape[1]
            tile_array = map_arr[tile_x_lower:tile_x_upper, tile_y_lower:tile_y_upper, :]
            tile_array = np.flipud(tile_array)
            tile_array = np.rot90(tile_array)
            tile_array_column.append(tile_array)
        tile_arrays.append(list(reversed(tile_array_column)))
    return list(reversed(tile_arrays))


def flipud_tiles_array(tiles_array):
    for x_value in range(len(tiles_array)):
        for y_value in range(len(tiles_array[0])):
            tiles_array[x_value][y_value] = np.flipud(tiles_array[x_value][y_value])
    return tiles_array


def rotate_tiles_array(tiles_array):
    for x_value in range(len(tiles_array)):
        for y_value in range(len(tiles_array[0])):
            tiles_array[x_value][y_value] = np.rot90(tiles_array[x_value][y_value])
    return tiles_array


# def select_tiles_slice(tiles_array, slice_coords):
#    x_coords, y_coords = slice_coords
#    x_lower, x_upper = x_coords
#    y_lower, y_upper = (len(tiles_array[0]) - y_coords[1], len(tiles_array[0]) - y_coords[0])
#    new_tiles_array = []
#    for x_value in range(x_lower, x_upper):
#        column = []
#        for y_value in range(y_lower, y_upper):
#            column.append(tiles_array[x_value][y_value])
#        new_tiles_array.append(column)
#    return list(reversed(new_tiles_array))

def create_image(tiles_array):
    img = np.concatenate([np.concatenate(tiles_column, axis=0) for tiles_column in tiles_array], axis=1)
    return img


def create_grid_slice_image(tile_arrays, slice_coords):
    x_coords, y_coords = slice_coords
    x_lower, x_upper = x_coords
    y_lower, y_upper = y_coords
    img_slice = []
    out_of_range_tile = np.zeros(shape=(8, 8, 3)).astype(np.uint8)
    for x_value in range(x_lower, x_upper + 1):
        column = []
        for y_value in range(y_lower, y_upper + 1):
            if x_value < 0 or y_value < 0 or x_value >= len(tile_arrays) or y_value >= len(tile_arrays[0]):
                column.append(out_of_range_tile)
            else:
                column.append(tile_arrays[x_value][y_value])
        column = np.concatenate(column, axis=1)
        img_slice.append(column)
    img_slice = np.concatenate(img_slice, axis=0)
    img_slice = np.flipud(img_slice)
    return img_slice


def render_img_slice(img_slice, rotations=3):
    for rotation in range(rotations):
        img_slice = np.rot90(img_slice)
    # img_slice = np.rot90(np.rot90(img_slice))
    img = Image.fromarray(img_slice, 'RGB')
    img.show()


def create_hiker():
    hiker_arr = np.zeros((8, 8, 3))
    color = (255, 255, 0)
    hiker_arr[1, 1] = color
    hiker_arr[1, 7] = color
    hiker_arr[2, 2] = color
    hiker_arr[2, 6] = color
    hiker_arr[3, 3:6] = color
    hiker_arr[4:8, 4] = color
    hiker_arr[6, 2:7] = color
    hiker_arr = np.flipud(hiker_arr)
    return hiker_arr.astype(np.uint8)


def create_craft(heading, altitude=3, color=(255, 0, 0)):
    craft_arr = np.zeros((8, 8, 3))
    if altitude == 3:
        craft_arr[1, 0:8] = color
        craft_arr[2:8, 3:5] = color
        craft_arr[6, 2:6] = color
        craft_arr[5, 1] = color
        craft_arr[5, 6] = color
        craft_arr[4, 0] = color
        craft_arr[4, 7] = color
    elif altitude == 2:
        craft_arr[1, 1:7] = color
        craft_arr[2:7, 3:5] = color
        craft_arr[5, 2:6] = color
        craft_arr[4, 1] = color
        craft_arr[4, 6] = color
    elif altitude == 1:
        craft_arr[1, 2:6] = color
        craft_arr[2:6, 3:5] = color
        craft_arr[4, 2:6] = color

    craft_arr = craft_arr.astype(np.uint8)
    rotate_dict = {0: 2, 1: 1, 2: 0, 3: 3}
    for i in range(rotate_dict[heading]):
        craft_arr = np.rot90(craft_arr)
    return craft_arr


def add_object(tile_arrays, object_arr, position):
    tile_arrays = deepcopy(tile_arrays)
    x_coord, y_coord = position
    tile_arrays[x_coord][y_coord] = np.maximum(tile_arrays[x_coord][y_coord], object_arr)
    return tile_arrays


def create_trace(hiker_position, position_sequence, label_sequence, lower_lon=0, upper_lon=499, lower_lat=0,
                 upper_lat=499, label_color_map={0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}):
    image_load_path = 'world-4096.png'
    tiles_array = create_tiles(image_load_path, (500, 500))
    tiles_array = flipud_tiles_array(tiles_array)
    tiles_array = rotate_tiles_array(tiles_array)
    valid_coords = ((lower_lon, upper_lon), (lower_lat, upper_lat))
    tiles_array = select_tiles_slice(tiles_array, valid_coords)
    hiker_arr = create_hiker()
    tiles_array = add_object(tiles_array, hiker_arr, hiker_position)
    for time_step in range(len(position_sequence)):
        x_coord, y_coord, altitude, heading = position_sequence[time_step]
        color = label_color_map[label_sequence[time_step]]
        craft_arr = create_craft(heading, altitude, color=color)
        tiles_array = add_object(tiles_array, craft_arr, (x_coord, y_coord))
    img = create_image(tiles_array)
    img = img.astype(np.uint8)
    img = np.flipud(img)
    return img


def select_tiles_slice(tiles_array, coords):
    x_coords, y_coords = coords
    x_lower, x_upper = x_coords
    y_lower, y_upper = y_coords
    new_tiles_array = []
    for x_value in range(x_lower, x_upper):
        column = []
        for y_value in range(y_lower, y_upper):
            column.append(tiles_array[x_value][y_value])
        new_tiles_array.append(column)
    return new_tiles_array


def create_trace(hiker_position, position_sequence, label_sequence, lower_lon=0, upper_lon=499, lower_lat=0,
                 upper_lat=499, label_color_map={0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}):
    root_path, _ = os.path.split(__file__)
    image_load_path = os.path.join(root_path, 'world-4096.png')
    tiles_array = create_tiles(image_load_path, (500, 500))
    valid_coords = ((lower_lon, upper_lon), (lower_lat, upper_lat))
    hiker_arr = create_hiker()
    tiles_array = add_object(tiles_array, hiker_arr, hiker_position)
    for time_step in range(len(position_sequence)):
        x_coord, y_coord, altitude, heading = position_sequence[time_step]
        color = label_color_map[label_sequence[time_step]]
        craft_arr = create_craft(heading, altitude, color=color)
        tiles_array = add_object(tiles_array, craft_arr, (x_coord, y_coord))
    tiles_array = select_tiles_slice(tiles_array, valid_coords)
    img = create_image(tiles_array)
    img = img.astype(np.uint8)
    img = np.flipud(img)
    return img


def test_recreate_map():
    image_file_path = 'world-4096.png'
    gridworld_dimensions = (500, 500)
    tiles_array = create_tiles(image_file_path, gridworld_dimensions)
    tiles_array = select_tiles_slice(tiles_array, ((260, 280), (270, 290)))
    img = create_image(tiles_array)
    render_img_slice(img, 2)


def test_create_trace1():
    hiker_position = (260, 270)
    position_sequence = [(265, 273, 3, 3), (263, 273, 2, 3), (262, 273, 2, 3), (261, 273, 2, 3), (261, 272, 2, 0),
                         (261, 270, 1, 0), (260, 270, 1, 3)]
    label_sequence = [0, 1, 1, 1, 1, 2, 2]
    lower_lon = 250
    upper_lon = 280
    lower_lat = 260
    upper_lat = 290
    img = create_trace(hiker_position=hiker_position, position_sequence=position_sequence,
                       label_sequence=label_sequence, lower_lon=lower_lon, upper_lon=upper_lon, lower_lat=lower_lat,
                       upper_lat=upper_lat)
    render_img_slice(img, 2)


def test_create_trace2():
    hiker_position = (412, 470)
    position_sequence = [[422, 460, 3, 1], [422, 461, 3, 2], [422, 462, 3, 2], [422, 463, 3, 2], [422, 464, 3, 2],
                         [422, 465, 3, 2], [422, 466, 3, 2], [421, 466, 3, 3], [420, 466, 3, 3], [420, 467, 3, 2],
                         [420, 468, 3, 2], [419, 468, 3, 3], [419, 469, 3, 2], [418, 469, 3, 3], [417, 469, 3, 3],
                         [416, 469, 3, 3], [414, 469, 2, 3], [412, 469, 1, 3], [412, 470, 1, 2]]
    label_sequence = [2, 1, 0, 0, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 0]
    lower_lon = 402
    upper_lon = 423
    lower_lat = 460
    upper_lat = 481
    img = create_trace(hiker_position=hiker_position, position_sequence=position_sequence,
                       label_sequence=label_sequence, lower_lon=lower_lon, upper_lon=upper_lon, lower_lat=lower_lat,
                       upper_lat=upper_lat)
    render_img_slice(img, 2)


def main():
    # image_file_path = 'world-4096.png'
    # gridworld_dimensions = (500, 500)
    # tile_arrays = create_tiles(image_file_path, gridworld_dimensions)
    # craft_arr = create_craft(2)
    # tile_arrays = add_object(tile_arrays, craft_arr, (0,488))
    # img_slice = create_grid_slice_image(tile_arrays, ((0, 11), (488, 499)))
    # render_img_slice(img_slice)
    test_recreate_map()
    test_create_trace2()


if __name__ == '__main__':
    main()