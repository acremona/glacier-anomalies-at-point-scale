import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

path_automatic_readings = 'C:\\Users\\Aaron\\Documents\\Holfuy\\2021\\automatic_reading\\1006_mT_Hist_2021-08-01_06-13_2021-09-29_09-08.csv'
path_manual_readings = 'C:\\Users\\Aaron\\Documents\\Holfuy\\2021\\1006_manual\\1006_manual.xlsx'
path_automatic_readings_mS = 'C:\\Users\\Aaron\\Documents\\Holfuy\\2021\\automatic_reading\\1006_mT_mS_dt2.xlsx'

auto_readings = pd.read_csv(path_automatic_readings)
manual_readings = pd.read_excel(path_manual_readings)
auto_readings_mS = pd.read_excel(path_automatic_readings_mS)
final = pd.read_csv('C:\\Users\\Aaron\\Documents\\Holfuy\\2021\\final_readings\\1006_abl_sum.csv')

#auto_readings_mS['rate'] = auto_readings_mS['rate'].rolling().mean()

start_year = int(path_automatic_readings.split('\\')[-1].split('_')[3].split('-')[0])
start_month = int(path_automatic_readings.split('\\')[-1].split('_')[3].split('-')[1])
start_day = int(path_automatic_readings.split('\\')[-1].split('_')[3].split('-')[2])
start_hour = int(path_automatic_readings.split('\\')[-1].split('_')[4].split('-')[0])
start_minute = int(path_automatic_readings.split('\\')[-1].split('_')[4].split('-')[1])

end_year = int(path_automatic_readings.split('\\')[-1].split('_')[5].split('-')[0])
end_month = int(path_automatic_readings.split('\\')[-1].split('_')[5].split('-')[1])
end_day = int(path_automatic_readings.split('\\')[-1].split('_')[5].split('-')[2])
end_hour = int(path_automatic_readings.split('\\')[-1].split('_')[6].split('-')[0])
end_minute = int(path_automatic_readings.split('\\')[-1].split('_')[6].split('-')[1].split('.')[0])


start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day, hour=start_hour, minute=start_minute)
end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day, hour=end_hour, minute=end_minute)

print(start_date, end_date)

initial_date_str = manual_readings['initial_date'].tolist()
final_date_str = manual_readings['final_date'].tolist()
manual_disp = manual_readings['displacements [cm ice]'].tolist()

# displacements mT_Hist
time_delta = auto_readings['time[h]'].tolist()
time = []
for t in time_delta:
    time.append(start_date + pd.Timedelta(hours=t))

auto_readings = pd.DataFrame({'date': time, 'dh': auto_readings['displacement rate [cm]'].tolist()}) #data=[[time,auto_readings['displacement rate [cm]'].tolist()]], columns=['time', 'displ'])

daily_displ = auto_readings.groupby(pd.Grouper(key='date', freq='1d')).sum().reset_index()

# displacements mT_mS
time_delta_mS = auto_readings_mS['time'].tolist()
time = []
for t in time_delta_mS:
    time.append(start_date + pd.Timedelta(hours=t))

print(len(time), len(auto_readings_mS['rate'].tolist()))
auto_readings_mS = pd.DataFrame({'date': time, 'dh': auto_readings_mS['rate'].tolist()}) #data=[[time,auto_readings['displacement rate [cm]'].tolist()]], columns=['time', 'displ'])

daily_displ_mS = auto_readings_mS.groupby(pd.Grouper(key='date', freq='1d')).sum().reset_index()

initial_date = []
final_date = []
periodic_sum = []
periodic_sum_mS = []
periodic_sum_fin = []

for idx, (init,fin) in enumerate(zip(initial_date_str, final_date_str)):
    temp_sum = 0
    temp_sum_mS = 0
    temp_sum_fin = 0
    print('init', init)
    y1 = int(init.split('.')[2])
    m1 = int(init.split('.')[1])
    d1 = int(init.split('.')[0])
    y2 = int(fin.split('.')[2])
    m2 = int(fin.split('.')[1])
    d2 = int(fin.split('.')[0])

    initial_date.append(pd.Timestamp(year=y1, month=m1, day=d1))
    final_date.append(pd.Timestamp(year=y2, month=m2, day=d2))

    interval = pd.Interval(pd.Timestamp(year=y1, month=m1, day=d1), pd.Timestamp(year=y2, month=m2, day=d2))
    print(interval)

    #print(init, fin)
    for date, disp in zip(daily_displ['date'].tolist(), daily_displ['dh'].tolist()):
        if date in interval:
            temp_sum += disp
        #if init <= date < fin:
        #    temp_sum += disp
    periodic_sum.append(temp_sum)

    for date_mS, disp_mS in zip(daily_displ_mS['date'].tolist(), daily_displ_mS['dh'].tolist()):
        if date_mS in interval:
            temp_sum_mS += disp_mS
        # if init <= date < fin:
        #    temp_sum += disp
    periodic_sum_mS.append(temp_sum_mS)

    for date_fin, disp_fin in zip(final['date'].tolist(), final['dh'].tolist()):
        date_fin = pd.Timestamp(date_fin)
        if date_fin in interval:
            temp_sum_fin += disp_fin
        # if init <= date < fin:
        #    temp_sum += disp
    periodic_sum_fin.append(temp_sum_fin)

mean_auto = np.divide((daily_displ['dh'].div(100) + daily_displ_mS['dh']), 2)

print(daily_displ['dh'].iloc[1])

print(periodic_sum)
print(manual_disp)
#plt.scatter(final_date, np.divide(periodic_sum, 100), label='automatic mT')
#plt.scatter(final_date, periodic_sum_mS, label='automatic mS')
plt.scatter(final_date, periodic_sum_fin, label='automatic')

#plt.scatter(final_date, manual_disp, label='manual')
plt.errorbar(final_date, np.divide(manual_disp,100), color='orange', yerr = 0.04, fmt='o', ecolor='black', label='manual')
plt.legend()
#plt.savefig('C:\\Users\\Aaron\\Documents\\Holfuy\\2021\\final_readings\\1006_2_weekly_comp.png')
plt.show()

plt.plot(daily_displ['date'].tolist(), np.cumsum(daily_displ['dh'].tolist()), label='automatic mT')
plt.plot(daily_displ_mS['date'].tolist(), np.cumsum(daily_displ_mS['dh'].tolist()), label='automatic mS')
#plt.plot(daily_displ_mS['date'].tolist(), np.cumsum(mean_auto), label='automatic mean')

#plt.plot(initial_date, np.cumsum(periodic_sum))
plt.scatter(final_date, np.cumsum(manual_disp), color='orange', label='manual')
plt.legend()
#plt.savefig('C:\\Users\\Aaron\\Documents\\Holfuy\\2021\\final_readings\\1006_2_cumulative_comp.png')
plt.show()
daily_displ['dh'] = daily_displ['dh'].div(100)
#daily_displ.to_csv('C:\\Users\\Aaron\\Documents\\Holfuy\\2021\\final_readings\\1006_2_abl_sum.csv', index=False)  # , columns=column_names)


plt.plot(daily_displ['date'].tolist(), daily_displ['dh'].tolist(), marker='.', label='automatic mT')
#plt.plot(daily_displ_mS['date'].tolist(), daily_displ_mS['dh'].tolist(), label='automatic mS')
#plt.plot(daily_displ['date'].tolist(), mean_auto, label='automatic mean')

#plt.plot(initial_date, np.cumsum(periodic_sum))
#plt.scatter(final_date, np.cumsum(manual_disp), color='orange', label='manual')
plt.legend()
#plt.savefig('C:\\Users\\Aaron\\Documents\\Holfuy\\2021\\final_readings\\1006_cumulative_comp.png')
plt.show()
print(np.max(np.cumsum(daily_displ['dh'].tolist())))
print(np.cumsum(manual_disp))
print(periodic_sum)


#
