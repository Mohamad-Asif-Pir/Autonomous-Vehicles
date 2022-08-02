#this code prints the carrla.location of manual control car every 2 secs. if no manual car then takes traffic or crashes with nothing found, so run manual control first
#some imports are extra
from __future__ import print_function
import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
client = carla.Client('10.100.12.113', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_lib=world.get_blueprint_library()

my_car = world.get_actors().filter('vehicle.*')[0]
x=10
for i in range(x):#does it x time. 
    print(my_car.get_location())
    time.sleep(2)