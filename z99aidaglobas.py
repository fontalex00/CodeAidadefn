
import numpy as np
import torch

# Globals read /write
epoch = 0
butch = 0
dloss = 0
dloss2 = 0
daccur = 0
butchsze = 0
lrsize = np.zeros( 5,dtype=int)
dpindo = np.zeros((12,5,2))
insel  = np.zeros((5,9))
grad1  = np.zeros( 5, dtype=np.object)
grad2  = np.zeros( 5, dtype=np.object)
grold  = np.zeros( 5, dtype=np.object)
whtcopy= np.zeros( 5, dtype=np.object)
gdmean = np.zeros( 5)
gxmean = np.zeros( 5)
nord   = torch.zeros((5,17,512))
ubias  = [0,0,0,0,0]


