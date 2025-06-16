'''
Goal: Find the points for wing extraction 
Input: wing filename 
Output: points for wing extraction 
'''

import deeplabcut as dlc


config_path='/Users/danyunhe/Desktop/wing-dana-2023-11-24/config.yaml'
dlc.analyze_time_lapse_frames(config_path,'filename',save_as_csv=True)

# somewhere to save the results? 
# input: folder with images, txt file of population id 