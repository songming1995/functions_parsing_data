import numpy as np
import os
from os.path import exists, join, isfile, dirname, abspath, split

import os, os.path
DIR = '/home/songming/Music/Kitti/testing/velodyne'
num_iter = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
path = '/home/songming/Music/Kitti/testing/velodyne'   

for i in range(num_iter):
    old_file_name = f'{i:010}'+'.bin'
    new_file_name = f'{i:06}'+'.bin'  
    old_file_name = join(path, old_file_name)
    new_file_name = join(path, new_file_name)
    
    os.rename(old_file_name, new_file_name)
    
    print("File renamed!")
