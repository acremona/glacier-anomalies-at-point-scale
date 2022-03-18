import mT_Hist
import numpy as np
import datetime
from datetime import timedelta
my_folder = "C:\\Users\\Aaron\\Documents\\Holfuy\\2021\\1008"
template = "C:\\Users\\Aaron\\Documents\\Holfuy\\2021\\templates\\template_1.PNG"
first_date = '2021-06-04_05-28'
end_date = '2021-06-17_17-37'

first_date_datetime = first_date.replace('_', '-')
time = list(map(int, first_date_datetime.split('.')[0].split('-')))                # remove file ending (eg. .jpg) and split string into a list of y, m, d, h, s
dt = datetime.datetime(time[0], time[1], time[2], time[3], time[4])

x, displacements, conversion_factors = mT_Hist.matchTemplate_hist(my_folder, template, 0.75, first_date, end_date, wait=1, vis=True, plotting=True, csv=True)

print(time, dt, np.max(x), dt + timedelta(hours = np.max(x)))
print(np.max(displacements))
