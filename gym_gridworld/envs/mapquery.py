'''To query the mavsim map'''


import sys
import socket
import struct

#import actr
import threading
import yaml

import uuid
import itertools

#use a tcp server


mavsim_server = ('127.0.0.1', 32786)
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#sent = sock.sendto(b'(\'TELEMETRY\', \'SET_MULTICAST\', \'ON\')', server_address)
#sent = send_sock.sendto(b'(\'TELEMETRY\', \'ADD_LISTENER\', \'127.0.0.1\', 9027, 0)', mavsim_server)

def terrain_request(lon=0,lat=0,width=5,height=5):
    startlat = lat
    startlon = lon
    list1 = list(range(startlon,startlon+width))
    list2 = list(range(startlat,startlat+height))

    #print(list1,list2)
    combinations = list(itertools.product(list1,list2))

    terrain_by_pair = []
    results_dict = {}
    #get the terrain at each coordinate from mavsim
    for pair in combinations:
        msg = "('{}', '{}', {}, {})".format('FLIGHT','MS_QUERY_TERRAIN',pair[1],pair[0])
        sent = send_sock.sendto(msg.encode('utf-8'),mavsim_server)
        data,address = send_sock.recvfrom(1024)
        data = data.decode('utf-8')
        tup_data = eval(data)
        if 'ERR' in tup_data:
            raise 'Error in mavsim'
        #print(tup_data)
        results_dict[(tup_data[2],tup_data[3])] = (tup_data[4],tup_data[5])


    return results_dict


