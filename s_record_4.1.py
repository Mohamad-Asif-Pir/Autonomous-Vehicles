import os
import time
import carla
import csv
import pygame
from datetime import datetime
#print("deleting")

cam_path1="C:\\Users\\Lenovo\\Downloads\\images\\out1\\output_cam1\\"
csv_path1="C:\\Users\\Lenovo\\Downloads\\images\\out1\\output_csv\\img.csv"

#for i in os.listdir(cam_path1):
    
    #os.remove(cam_path1+i)
    
#for i in os.listdir(csv_path1):
    #os.remove(csv_path1+i)

record_time=1000

print("recording")

client = carla.Client('10.100.12.113', 2000)
client.set_timeout(10.0)
world = client.get_world()
map=world.get_map()
blueprint_lib=world.get_blueprint_library()

for i in world.get_actors():
    if(i.type_id=='traffic.traffic_light'):
        i.set_state(carla.TrafficLightState.Green)
        i.set_green_time(record_time+100)
my_car = world.get_actors().filter('vehicle.*')[0]

camera_bp_ct=blueprint_lib.find("sensor.camera.rgb")
camera_bp_lt=blueprint_lib.find("sensor.camera.rgb")
camera_bp_rt=blueprint_lib.find("sensor.camera.rgb")

camera_bp_ct.set_attribute('sensor_tick', '0.1')
camera_bp_lt.set_attribute('sensor_tick', '0.1')
camera_bp_rt.set_attribute('sensor_tick', '0.1')

relative_transform=carla.Transform(carla.Location(x=1.6,y=0,z=1.5))
camera_ct = world.spawn_actor(camera_bp_ct, relative_transform, attach_to=my_car)
relative_transform=carla.Transform(carla.Location(x=1.6,y=-1.3,z=1.5))
camera_lt = world.spawn_actor(camera_bp_lt, relative_transform, attach_to=my_car)
relative_transform=carla.Transform(carla.Location(x=1.6,y=1.3,z=1.5))
camera_rt = world.spawn_actor(camera_bp_rt, relative_transform, attach_to=my_car)

csv_data_ct=[]
csv_data_lt=[]
csv_data_rt=[]
def save_cam_ct(image):
    now = datetime.now()
    current_time = now.strftime("%H%M%S")
    image.save_to_disk(cam_path1+current_time+'ct%06d.png' % image.frame)
    ctrl=my_car.get_control()
    direction=4
    #1 left
    #2 go straight
    #3 right
    #4 lane follow
    if(map.get_waypoint(my_car.get_location(),project_to_road=True,lane_type=carla.LaneType.Driving).is_junction):
        lights=my_car.get_light_state()
        direction=4
        if(lights==carla.VehicleLightState.LeftBlinker):
            direction=1
        elif(lights==carla.VehicleLightState.RightBlinker):
            direction=3
        else:
            direction=2
    tmp=[current_time+'ct%06d.png' % image.frame,ctrl.throttle,ctrl.steer,direction]
    csv_data_ct.append(tmp)
    

def save_cam_lt(image):
    now = datetime.now()
    current_time = now.strftime("%H%M%S")
    image.save_to_disk(cam_path1+current_time+'lt%06d.png' % image.frame)
    ctrl=my_car.get_control()
    direction=4
    if(map.get_waypoint(my_car.get_location(),project_to_road=True,lane_type=carla.LaneType.Driving).is_junction):
        lights=my_car.get_light_state()
        direction=4
        if(lights==carla.VehicleLightState.LeftBlinker):
            direction=1
        elif(lights==carla.VehicleLightState.RightBlinker):
            direction=3
        else:
            direction=2
    tmp=[current_time+'lt%06d.png' % image.frame,ctrl.throttle,ctrl.steer,direction]
    csv_data_lt.append(tmp)

def save_cam_rt(image):
    now = datetime.now()
    current_time = now.strftime("%H%M%S")
    image.save_to_disk(cam_path1+current_time+'rt%06d.png' % image.frame)
    ctrl=my_car.get_control()
    direction=4
    if(map.get_waypoint(my_car.get_location(),project_to_road=True,lane_type=carla.LaneType.Driving).is_junction):
        lights=my_car.get_light_state()
        direction=4
        if(lights==carla.VehicleLightState.LeftBlinker):
            direction=1
        elif(lights==carla.VehicleLightState.RightBlinker):
            direction=3
        else:
            direction=2
    tmp=[current_time+'rt%06d.png' % image.frame,ctrl.throttle,ctrl.steer,direction]
    csv_data_rt.append(tmp)

camera_ct.listen(lambda image: save_cam_ct(image))
camera_lt.listen(lambda image: save_cam_lt(image))
camera_rt.listen(lambda image: save_cam_rt(image))

time.sleep(record_time)

print("done recording")
filename = csv_path1
with open(filename, 'a',newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(csv_data_ct)
    csvwriter.writerows(csv_data_lt)
    csvwriter.writerows(csv_data_rt)
camera_ct.destroy()
camera_lt.destroy()
camera_rt.destroy()