import numpy as np
import cv2
import os
import datetime
import math
import imutils
import xlsxwriter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matchtemplate_funct import match_template
from Meanshift_funct import mean_shift_diff_frames
from meanshift_same_frame import mean_shift_same_frame
from manual_mask import create_red_mask,create_yellow_mask,create_blue_mask,create_green_mask, create_black_mask
from sklearn.linear_model import LinearRegression
import pylab

max_val_idx = 88

#datetime vector
start_date = datetime.date(2019, 6, 28)
start_min = datetime.datetime(2019, 6, 27, 11, 59)
temp_date = start_date
delta_day = datetime.timedelta(days=1)


# import data from excel
data = pd.read_excel("C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\1001.xlsx",sheet_name='Final')
time_val = data['time val'].tolist()
time_val = time_val[0:max_val_idx]
rate_val = data['rate val'].tolist()
dy_cum_val = data['dy cum val'].tolist()
dy_cum_val = dy_cum_val[0:max_val_idx]
time = data['time']
dy_cum1 = data['dy cum1']
#dy_cum2 = data['dy cum2']
dy_cum3 = data['dy cum3']
dy_mean = data['mean']
std1 = data[' std dev cum1'].tolist()
#std2 = data[' std dev cum2'].tolist()
std3 = data[' std dev cum3'].tolist()
stdmean = data['mean std']

dates = []
dates_min = []

for t in time_val:
    dates.append(temp_date)
    temp_date += delta_day

for t1 in time:
    delta_min = datetime.timedelta(hours=t1)
    dates_min.append(start_min+delta_min)

#plot
fig, ax = plt.subplots()
plt.figure(figsize=(8,4))
plt.plot(dates, dy_cum_val, '.', color= 'black', label='Validation data')
plt.plot(dates_min,dy_cum1,'-', color='blue', label='Template 1')
#plt.plot(dates_min,dy_cum2,'-', color='purple', label='Template 2')
plt.plot(dates_min,dy_cum3,'-', color='limegreen', label='Template 3')
plt.plot(dates_min,dy_mean,'-', color='red', label='Mean',linewidth=2)
#plt.fill_between(dates_min,dy_cum1-std1, dy_cum1+std1,color='blue', alpha=0.2)
#plt.fill_between(dates_min,dy_cum2-std2, dy_cum2+std2,color='orange', alpha=0.2)
#plt.fill_between(dates_min,dy_cum3-std3, dy_cum3+std3,color='green', alpha=0.2)
plt.fill_between(dates_min,dy_mean-stdmean, dy_mean+stdmean,color='red', alpha=0.15)
plt.gcf().autofmt_xdate()

# Format the date into months & days
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

# Change the tick interval
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))

plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)

plt.xlabel('Time [-]',fontsize='large', fontweight= 'heavy') #do it in datetime
plt.ylabel('Cumulative displacement [m]',fontsize='large',fontweight= 'heavy')

plt.legend(loc='upper left', shadow=False, fontsize='medium')
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.savefig('1001.png')
plt.show()



