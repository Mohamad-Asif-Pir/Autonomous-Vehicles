import os
import csv
import carla
from carla import ColorConverter as cc
import numpy as np
print("deleting")
cam_path1="C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_cam\\"
seg_path1="C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_seg\\"
seg_path2="C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_seg2\\"
csv_path1="C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_csv\\img_seg_ctrl.csv"
####################
# cam_path1="/home/vinayakpc/PythonAPI/examples/out/output_cam"
# seg_path1="/home/vinayakpc/PythonAPI/examples/out/output_seg"
# seg_path2="/home/vinayakpc/PythonAPI/examples/out/output_seg2"
# csv_path1="/home/vinayakpc/PythonAPI/examples/out/output_csv/img_seg_ctrl.csv"

for i in os.listdir(seg_path2):
    os.remove(seg_path2+i)

for i in os.listdir(seg_path1):
    os.remove(seg_path1+i)

for i in os.listdir(cam_path1):
    os.remove(cam_path1+i)

print("recording")

client = carla.Client('10.100.12.113', 2000)
client.set_timeout(10.0)
world = client.get_world()
map=world.get_map()
blueprint_lib=world.get_blueprint_library()

my_car = world.get_actors().filter('vehicle.*')[0]

camera_bp_ct=blueprint_lib.find("sensor.camera.rgb")
camera_bp_lt=blueprint_lib.find("sensor.camera.rgb")
camera_bp_rt=blueprint_lib.find("sensor.camera.rgb")
#seg_camera_bp=blueprint_lib.find("sensor.camera.semantic_segmentation")

camera_bp_ct.set_attribute('sensor_tick', '0.1')
camera_bp_lt.set_attribute('sensor_tick', '0.1')
camera_bp_rt.set_attribute('sensor_tick', '0.1')
#seg_camera_bp.set_attribute('sensor_tick', '0.1')

relative_transform=carla.Transform(carla.Location(x=1.6,y=0,z=1.5))
camera_ct = world.spawn_actor(camera_bp_ct, relative_transform, attach_to=my_car)
relative_transform=carla.Transform(carla.Location(x=1.6,y=-1,z=1.5))
camera_lt = world.spawn_actor(camera_bp_lt, relative_transform, attach_to=my_car)
relative_transform=carla.Transform(carla.Location(x=1.6,y=1,z=1.5))
camera_rt = world.spawn_actor(camera_bp_rt, relative_transform, attach_to=my_car)
#seg_camera = world.spawn_actor(seg_camera_bp, relative_transform, attach_to=my_car)

csv_data=[]
def save_cam_nor(image):
    image.save_to_disk(cam_path1+'/%06d.png' % image.frame)
    ctrl=my_car.get_control()
    direction=4
    #1 left
    #2 go straight
    #3 right
    #4 lane follow
    if(map.get_waypoint(my_car.get_location(),project_to_road=True,lane_type=carla.LaneType.Driving).is_junction):
        if(ctrl.steer<0):
            direction=1
        elif(ctrl.steer>0):
            direction=3
        else:
            direction=2
    tmp=[image.frame,ctrl.throttle,ctrl.steer *4,direction]
    csv_data.append(tmp)


def save_cam_seg(image1):
    #print("seg : "+str(image1.frame)+" "+str(image1.timestamp))
    image1.save_to_disk(seg_path2+'/%06d.png' % image1.frame)
    image1.save_to_disk(seg_path1+'/%06d.png' % image1.frame,color_converter=carla.ColorConverter.CityScapesPalette)
    # ctrl=my_car.get_control()
    # direction=4
    # #1 left
    # #2 go straight
    # #3 right
    # #4 lane follow
    # if(map.get_waypoint(my_car.get_location(),project_to_road=True,lane_type=carla.LaneType.Driving).is_junction):
    #     if(ctrl.steer<0):
    #         direction=1
    #     elif(ctrl.steer>0):
    #         direction=3
    #     else:
    #         direction=2
    # tmp=[image1.frame,ctrl.throttle,ctrl.steer,direction]
    # csv_data.append(tmp)

camera.listen(lambda image: save_cam_nor(image))
#seg_camera.listen(lambda image1: save_cam_seg(image1))


time.sleep(7200)

print("done recording")
filename = csv_path1
with open(filename, 'w',newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(csv_data)
camera.destroy()
#seg_camera.destroy()