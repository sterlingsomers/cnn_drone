from absl import flags
from json import dumps as dump_as_json_string
from pprint import pformat

FLAGS = flags.FLAGS

original_drawn_map = [
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 26,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 26, 26,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2, 26, 26, 25,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2, 26, 26, 25,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2, 26,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
]

default_drawn_map_key = 'original_drawn_map'
default_hiker_location = (4,4)
default_drone_location = (17,17)

long_mountain_map = [
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 26, 26,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 26, 26,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 26, 26,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 26,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 26, 26,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2, 26, 26, 25,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2, 26, 26, 25,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2, 26,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
]

hiker_by_mountain_map = [
     [3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [3,  3,  3, 26,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 26,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 26, 26,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2, 26, 26, 25,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2, 26, 26, 25,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2, 26,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
]


map_dictionary = {
     'original_drawn_map':  original_drawn_map,
     'long_mountain_map': long_mountain_map,
     'hiker_by_mountain': hiker_by_mountain_map,
}

def get_map():
     print('Using map ' + str(FLAGS.map))
     map_name = FLAGS.map
     if map_name in map_dictionary:
          return map_dictionary[map_name]
     return map_dictionary[default_drawn_map_key]


def get_hiker():
     try:
          print('Using hiker (' + str(FLAGS.hikerx) + ',' + str(FLAGS.hikery) + ')')
          return (FLAGS.hikerx,FLAGS.hikery)
     except Exception as e:
          print('Error Using hiker (' + str(default_hiker_location[0]) + ',' + str(default_hiker_location[1]) + ')'+ ":" + str(e))
          return default_hiker_location


def get_drone():
     try:
          print('Using drone (' + str(FLAGS.dronex) + ',' + str(FLAGS.droney) + ')')
          return (FLAGS.dronex,FLAGS.droney)
     except Exception as e:
          print('Error Using drone (' + str(default_drone_location[0]) + ',' + str(default_drone_location[1]) + ')' + ":" + str(e))
          return default_drone_location


def get_map_names():
     return dump_as_json_string(list(map_dictionary.keys()))


def get_map_by_name(name):
     col_width = 3
     if name in map_dictionary:
          matrix_string = ""
          for row in map_dictionary[name]:
               for element in row:
                    matrix_string += "".join(str(element).rjust(col_width))
               matrix_string += '\n'
          return matrix_string
     return []

study_help_keys = ["map", "hikerx", "hikery", "dronex", "droney", "getnames", "getmap", "studyhelp"]

def studyhelp():
     pprintstring = ''
     for key in study_help_keys:
          padkey = key + ":"
          pprintstring += padkey.ljust(9) +  (FLAGS[key].help) + '\n'
     return pprintstring
