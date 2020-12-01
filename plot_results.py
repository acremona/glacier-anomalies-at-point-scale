import numpy as np
import cv2
import os
import datetime
import math
import imutils
import xlsxwriter
import pandas as pd
import matplotlib.pyplot as plt
from matchtemplate_funct import match_template
from Meanshift_funct import mean_shift_diff_frames
from meanshift_same_frame import mean_shift_same_frame
from manual_mask import create_red_mask,create_yellow_mask,create_blue_mask,create_green_mask, create_black_mask
from sklearn.linear_model import LinearRegression
import pylab

#fare dates su x axis


# import data from excel
data = pd.read_excel("C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\1008.xlsx",sheet_name='Final')
time_val = data['time val'].tolist()
rate_val = data['rate val'].tolist()
dy_cum_val = data['dy cum val'].tolist()
time = data['time']
dy_cum1 = data['dy cum1']
dy_cum2 = data['dy cum2']
dy_cum3 = data['dy cum3']
dy_mean = data['mean']
std1 = data[' std dev cum1'].tolist()
std2 = data[' std dev cum2'].tolist()
std3 = data[' std dev cum3'].tolist()

#plot
plt.plot(time_val, dy_cum_val, 'or', label='Validation data')
plt.plot(time,dy_cum1,'-', color='blue', label='Template 1')
plt.plot(time,dy_cum2,'-', color='orange', label='Template 2')
plt.plot(time,dy_cum3,'-', color='green', label='Template 3')
plt.plot(time,dy_mean,'-', color='black', label='Mean',linewidth=2)
#plt.fill_between(time,dy_cum1-std1, dy_cum1+std1,color='blue', alpha=0.2)
#plt.fill_between(time,dy_cum2-std2, dy_cum2+std2,color='orange', alpha=0.2)
#plt.fill_between(time,dy_cum3-std3, dy_cum3+std3,color='green', alpha=0.2)
plt.fill_between(time,dy_mean-std3, dy_mean+std3,color='gray', alpha=0.2)
plt.xlabel('Time [-]',fontsize='x-large') #do it in datetime
plt.ylabel('Cumulative displacement [m]',fontsize='x-large')
plt.legend(loc='upper left', shadow=False, fontsize='x-large')
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.show()



