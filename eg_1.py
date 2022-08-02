


import carla
client=carla.Client("10.100.12.113",2000)
client.set_timeout(10.0)
world=client.load_world('Town03_Opt')