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
from pylab import *

def aggregate_2_daily(time, time_val, dy_cum_min):
    daily_dy = []
    daily_rate = []

    for tv in time_val:
        idx = np.where(time == min(time, key=lambda x: abs(x - tv)))
        daily_dy.append(dy_cum_min[idx])

    for i, d in enumerate(daily_dy):
        if i == 0:
            daily_rate.append(d)
        else:
            daily_rate.append(d-daily_dy[i-1])
    return daily_dy, daily_rate

##################################
max_val_idx = 52
threshold = -0.002

#datetime vector
start_date = datetime.date(2019, 8, 13)
start_min = datetime.datetime(2019, 8, 12, 17, 10)
temp_date = start_date
delta_day = datetime.timedelta(days=1)
data = pd.read_excel("C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\1006.xlsx",sheet_name='Final')
###################################

# import data from excel

time_val = data['time val'].tolist()
time_val = time_val[0:max_val_idx]
rate_val = data['rate val'].tolist()
rate_val = rate_val[0:max_val_idx]
dy_cum_val = data['dy cum val'].tolist()
dy_cum_val = dy_cum_val[0:max_val_idx]
time = data['time']
rate1 = data['rate1']
rate2 = data['rate2']
rate3 = data['rate3']
rate1 = np.where(rate1 > threshold, rate1, 0)
rate2 = np.where(rate2 > threshold, rate2, 0)
rate3 = np.where(rate3 > threshold, rate3, 0)
dy_cum_1_thresh = np.cumsum(rate1)
dy_cum_2_thresh = np.cumsum(rate2)
dy_cum_3_thresh = np.cumsum(rate3)
dy_mean_thresh = (dy_cum_1_thresh+dy_cum_2_thresh+dy_cum_3_thresh)/3

dy_cum1 = data['dy cum1']
dy_cum2 = data['dy cum2']
dy_cum3 = data['dy cum3']
dy_mean = data['mean']

std1 = data[' std dev cum1'].tolist()
std2 = data[' std dev cum2'].tolist()
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

daily_dy_1, daily_rate1 = aggregate_2_daily(time, time_val, dy_cum_1_thresh)
daily_dy_2, daily_rate2 = aggregate_2_daily(time, time_val, dy_cum_2_thresh)
daily_dy_3, daily_rate3 = aggregate_2_daily(time, time_val, dy_cum_3_thresh)
daily_dy_mean, daily_rate_mean = aggregate_2_daily(time, time_val, dy_mean_thresh)
daily_rate1 = [np.asscalar(np.nan_to_num(i)) for i in daily_rate1]
daily_rate2 = [np.asscalar(np.nan_to_num(i)) for i in daily_rate2]
daily_rate3 = [np.asscalar(np.nan_to_num(i)) for i in daily_rate3]
daily_rate_mean = [np.asscalar(np.nan_to_num(i)) for i in daily_rate_mean]


err1 = 100*(np.reshape(np.asarray(daily_rate1),(max_val_idx,1))-np.reshape(np.asarray(rate_val),(max_val_idx,1)))
err1 = np.reshape([np.asscalar(np.nan_to_num(i)) for i in err1],(max_val_idx))
err2 = 100*(np.reshape(np.asarray(daily_rate2),(max_val_idx,1))-np.reshape(np.asarray(rate_val),(max_val_idx,1)))
err2 = np.reshape([np.asscalar(np.nan_to_num(i)) for i in err2],max_val_idx)
err3 = 100*(np.reshape(np.asarray(daily_rate3),(max_val_idx,1))-np.reshape(np.asarray(rate_val),(max_val_idx,1)))
err3 = np.reshape([np.asscalar(np.nan_to_num(i)) for i in err3],max_val_idx)
errm = 100*(np.reshape(np.asarray(daily_rate_mean),(max_val_idx,1))-np.reshape(np.asarray(rate_val),(max_val_idx,1)))
errm = np.reshape([np.asscalar(np.nan_to_num(i)) for i in errm],max_val_idx)

errors = [err1, err2, err3, errm]
med = np.median(errors,axis=1)
med = [round(i,2) for i in med]

# PLOT
fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

# get dictionary returned from boxplot
labels = ['Template 1', 'Template 2', 'Template 3', 'Mean']

bp_dict = axs[1].boxplot(errors, vert=True, patch_artist=True, showfliers=False,labels=labels,widths=(0.2, 0.2, 0.2, 0.2))
axs[1].set_ylabel('Daily errors [cm]')

for i, line in enumerate(bp_dict['medians']):
    # get position data for median line
    x, y = line.get_xydata()[1] # top of median line
    # overlay median value
    text(x, y, med[i], verticalalignment='center') # draw above, centered

# fill with colors
colors = ['blue', 'purple', 'limegreen', 'red']
for i, box in enumerate(bp_dict['boxes']):
    # change outline color
    box.set(color=colors[i], linewidth=2, alpha=0.9)
    # change fill color
    box.set(facecolor = colors[i], alpha=0.3)
axs[1].grid(axis='y')


axs[0].plot(dates, dy_cum_val, '.', color= 'black', label='Validation data')
#plt.plot(dates_min,dy_cum1,'-', color='blue', label='Template 1')
axs[0].plot(dates_min,dy_cum_1_thresh,'-', color='blue', label='Template 1')
#plt.plot(dates_min,dy_cum2,'-', color='purple', label='Template 2')
axs[0].plot(dates_min,dy_cum_2_thresh,'-', color='purple', label='Template 2')
#plt.plot(dates_min,dy_cum3,'-', color='limegreen', label='Template 3')
axs[0].plot(dates_min,dy_cum_3_thresh,'-', color='limegreen', label='Template 3')
axs[0].plot(dates_min,dy_mean_thresh,'-', color='red', label='Mean')
#plt.plot(dates_min,dy_mean,'-', color='red', label='Mean',linewidth=2)
#plt.fill_between(dates_min,dy_cum1-std1, dy_cum1+std1,color='blue', alpha=0.2)
#plt.fill_between(dates_min,dy_cum2-std2, dy_cum2+std2,color='orange', alpha=0.2)
#plt.fill_between(dates_min,dy_cum3-std3, dy_cum3+std3,color='green', alpha=0.2)
axs[0].fill_between(dates_min,dy_mean_thresh-stdmean, dy_mean_thresh+stdmean,color='red', alpha=0.15)

for label in axs[0].get_xticklabels():
    label.set_ha("right")
    label.set_rotation(18)
# Format the date into months & days
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

# Change the tick interval
axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=5))

#axs[0].rc('xtick', labelsize=25)
#axs[0].rc('ytick', labelsize=25)

axs[0].set_ylabel('Displacement [m]')#,fontsize='large',fontweight= 'heavy')

axs[0].legend(loc='upper left', shadow=False, fontsize='small')
axs[0].grid(axis='y', color='gray', linestyle=':', linewidth=0.5)
plt.savefig('1006_w_bp.png')
plt.show()

###################################################################
writer = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')
df = pd.DataFrame({'time': time, 'min rate1': rate1, 'min rate2': rate2, 'min rate3': rate3, 'dy cum 1':dy_cum_1_thresh, 'dy cum 2':dy_cum_2_thresh, 'dy cum 3':dy_cum_3_thresh, 'dy cum mean':dy_mean_thresh})
df.to_excel(writer, sheet_name='Data')
df1 = pd.DataFrame({'d rate1':daily_rate1, 'd rate2':daily_rate2, 'd rate3':daily_rate3})
df1.to_excel(writer, sheet_name='Daily data')
###################################################################
writer.save()
