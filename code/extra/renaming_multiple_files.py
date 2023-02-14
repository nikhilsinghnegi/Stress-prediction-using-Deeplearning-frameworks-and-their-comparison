import shutil
import numpy as np
import os

data_set = 2000


for i in range(data_set):
    j = str(i+1)
    new = 'One_Stress_cart_' + j
    new_name    = 'D:\\8th sem\\BTP_II\\Compressed\\' + new + '.dat'
    
    old = 'Composite_uniform_Stress_Cart_' + j
    old_name    = 'D:\\8th sem\\BTP_II\\Compressed\\' + old + '.dat'
    
    destination_path = 'D:\\for_dat_file'
    os.rename(old_name, new_name)