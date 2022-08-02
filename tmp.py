import carla

client = carla.Client('10.100.12.113', 2000)
client.set_timeout(10.0)
world = client.get_world()
map=world.get_map()
blueprint_lib=world.get_blueprint_library()
for i in world.get_actors():
    if(i.type_id=='traffic.traffic_light'):
        i.set_state(carla.TrafficLightState.Green)
        i.set_green_time(3700)