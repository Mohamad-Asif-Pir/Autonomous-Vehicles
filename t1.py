import os


import csv
import subprocess

#for i in os.listdir('C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_cam'):
    #if i not in os.listdir('C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_seg'):
        #os.remove('C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_cam\\'+i)

data_dir='C:\\Users\\Lenovo\\Downloads\\images\\out1\\output_cam'
csv_path='C:\\Users\\Lenovo\\Downloads\\images\\out1\\output_csv\\img.csv'
#subprocess.run(["data_dir"],shell=True)
#csv_file = csv.reader(open('csv_path', "r"))
#csv_name_list=os.listdir(csv_path)
#with open(csv_path, 'r') as file:
#    csvreader = csv.reader(file)
    

for i in os.listdir('C:\\Users\\Lenovo\\Downloads\\images\\out1\\output_cam'):
    with open(csv_path, 'r') as file:
        flag=False
        csvreader = csv.reader(file)
        for row in csvreader:
            if i == row[0]:
                flag=True;
                break;
        if(flag==False):
            os.remove('C:\\Users\\Lenovo\\Downloads\\images\\out1\\output_cam\\'+i)
        
#data_dir='C:\\Users\\Lenovo\\Downloads\\images\\out\\output_cam'
#csv_path='C:\\Users\\Lenovo\\Downloads\\images\\out\\output_csv\\img.csv'
        
#csv_name_list=os.listdir(csv_path)
#with open(csv_path, 'r') as file:
    #csvreader = csv.reader(file)
    #for row in csvreader:
        #if(row[0]+'.png' not in img_name_list):
            #os.remove('C:\\Users\\Lenovo\\Downloads\\images\\out\\output_cam\\row[0]+.png')