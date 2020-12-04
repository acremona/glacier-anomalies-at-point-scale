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
from sklearn.linear_model import LinearRegression
import pylab
from matplotlib.backends.backend_agg import FigureCanvasAgg

max_val_idx = 88

#datetime vector
start_date = datetime.date(2019, 6, 27)
start_minutes = [datetime.datetime(2019, 6, 27, 11, 59), datetime.datetime(2019, 8, 1, 6, 11), datetime.datetime(2019, 6, 20, 5, 15), datetime.datetime(2019, 8, 13, 16, 50), datetime.datetime(2019, 7, 26, 11, 20), datetime.datetime(2019, 6, 27, 18, 31), datetime.datetime(2019, 8, 27, 6, 35)]
sheets = ["1001_plot", "1002_plot", "1003_plot", "1006_plot", "1007_plot", "1008_plot", "1009_plot"]
fails = [0, 2, 4, 0, 0, 5, 0]
temp_date = start_date
delta_day = datetime.timedelta(days=1)

for counter in range(len(sheets)):
    # import data from excel
    data = pd.read_excel("C:\\Users\\joelb\\OneDrive\\Dokumente\\sensitivity_smooth.xlsx",sheet_name=sheets[counter])
    time_val = data['time val'].tolist()
    dy_cum_val = data['dy_cum_val'].tolist()
    dy_cum_val = [a/100 for a in dy_cum_val]
    dy_cum = []
    for a in range(11):
        dy_cum.append(data['dy_cum'+str(a+1)].tolist())

    for a in range(len(dy_cum)):
        for b in range(len(dy_cum[a])):
            dy_cum[a][b] = dy_cum[a][b]/100

    dates = []
    dates_min = []

    for t in time_val:
        dates.append(start_minutes[counter] + datetime.timedelta(hours=t))

    #plot
    colors = ["#ff0000", "#e50919", "#cc1233", "#b21b4c", "#992466", "#7f2d7f", "#653699", "#4c3fb2", "#3248cc", "#1951e5", "#005bff"]

    n_lines = 11
    n_fail = fails[counter]
    errors = []

    for a in range(11-n_fail):
        error_thresh = []
        for i in range(len(dates)-3):
            daily1 = dy_cum_val[i+3]-dy_cum_val[i+2]
            daily2 = dy_cum[a][i+3]-dy_cum[a][i+2]
            error_thresh.append((daily2-daily1)*100)
        errors.append(error_thresh)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1]})
    labels = [a/100 for a in range(65, 86-2*n_fail, 2)]
    bp_dict = axs[1].boxplot(errors, vert=True, patch_artist=True, showfliers=False,labels=labels,widths=[0.2 for a in range(11-n_fail)])
    axs[1].set_ylabel('Daily errors [cm]',fontsize='medium', fontweight= 'heavy')
    axs[1].set_xlabel('Threshold [-]',fontsize='medium', fontweight= 'heavy')
    med = np.median(errors,axis=1)
    med = [round(i, 2) for i in med]

    for i, line in enumerate(bp_dict['medians']):
        # get position data for median line
        x, y = line.get_xydata()[1] # top of median line
        # overlay median value
        plt.text(x, y, med[i], verticalalignment='center') # draw above, centered

    # fill with colors
    for i, box in enumerate(bp_dict['boxes']):
        # change outline color
        box.set(color=colors[i], linewidth=2, alpha=0.9)
        # change fill color
        box.set(facecolor = colors[i], alpha=0.3)
    axs[1].grid(axis='y')

    for a in range(11):
        axs[0].plot(dates[:len(dy_cum[a])],dy_cum[a], color=colors[a], label='thresh = '+str(round(0.65+0.2/10*a, 2)), linewidth=0.8)
    axs[0].plot(dates, dy_cum_val, '.', color= 'orange', label='Validation data')

    for label in axs[0].get_xticklabels():
        label.set_ha("right")
        label.set_rotation(18)
    # Format the date into months & days
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

    # Change the tick interval
    axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    axs[0].set_ylabel('Displacement [m]',fontsize='medium', fontweight= 'heavy')
    axs[0].grid(color='gray', linestyle=':', linewidth=0.5)

    plt.show()
