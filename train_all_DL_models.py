'''
File to train all models from DL/experiments
'''
import os
import sys
from train_DL_1D import main as main_1d
from train_DL_2D import main as main_2d
from train_fusion import main as main_fusion

in_path_prefix = 'DL/experiments/'
elements = os.listdir(in_path_prefix)

for elem in elements:
    full_path = os.path.join(in_path_prefix, elem)

    if '1D' in elem:
        print('1D model', elem)
        sys.argv = ['train_DL_1D.py', '--config', full_path]
        try:
            main_1d()
        except:
            print('did not work')

    elif 'fusion' not in elem:
        print('2D model', elem)
        sys.argv = ['train_DL_2D.py', '--config', full_path]
        try:
            main_2d()
        except:
            print('did not work')

sys.argv = ['train_fusion.py', '--config', 'DL/runs/fusion_1D_unet_2D_res_attn_32']
main_fusion()