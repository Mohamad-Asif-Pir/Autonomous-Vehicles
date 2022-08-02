from __future__ import print_function
import glob
import os
import sys

import carla

client = carla.Client('10.100.12.113', 2000)
client.set_timeout(10.0)
#world = client.load_world('Town10HD')
world = client.load_world('Town03')
#world = client.reload_world()
#world = client.load_world('Town05')
#print(client.get_available_maps())
#['/Game/Carla/Maps/Town01', '/Game/Carla/Maps/Town01_Opt', '/Game/Carla/Maps/Town02', '/Game/Carla/Maps/Town02_Opt', '/Game/Carla/Maps/Town03', 
# '/Game/Carla/Maps/Town03_Opt', '/Game/Carla/Maps/Town04', '/Game/Carla/Maps/Town04_Opt', '/Game/Carla/Maps/Town05', '/Game/Carla/Maps/Town05_Opt', 
# '/Game/Carla/Maps/Town10HD', '/Game/Carla/Maps/Town10HD_Opt']