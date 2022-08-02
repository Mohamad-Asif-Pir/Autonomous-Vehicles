import os
pwd1=os.getcwd()
cam_path=pwd1+"\\out\\output_cam\\"
seg_path1=pwd1+"\\out\\output_seg\\"
seg_path2=pwd1+"\\out\\output_seg2\\"

print(len(os.listdir(cam_path)))
print(len(os.listdir(seg_path1)))

for i in os.listdir(cam_path):
    if i not in os.listdir(seg_path1):
        os.remove(cam_path+i)

for i in os.listdir(seg_path1):
    if i not in os.listdir(cam_path):
        os.remove(seg_path1+i)

for i in os.listdir(seg_path2):
    if i not in os.listdir(cam_path):
        os.remove(seg_path2+i)