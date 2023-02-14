import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
print(os.environ["CUDA_DEVICE_ORDER"])
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

#import pprint  #TO PRINT ENVIRONMET VARIABLES
#env_var = os.environ  #TO PRINT ENVIRONMET VARIABLES
#pprint.pprint(dict(env_var),width = 1)   #TO PRINT ENVIRONMET VARIABLES

