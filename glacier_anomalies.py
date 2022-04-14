import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os


def import_climatology(path):
    head = pd.read_csv(
        path, sep=';', skiprows=2, skipinitialspace=True,
        header=0, nrows=0, encoding='latin1').columns

    usecols = np.arange(len(head))
    colnames = head
    climatology = pd.read_csv(path,
                           skiprows=4, sep=' ', skipinitialspace=True,
                           usecols=usecols, header=None,
                           names=colnames, dtype={'date_s': str,
                                                  'date_f': str,
                                                  'date0': str,
                                                  'date1': str})
    climatology.columns = climatology.columns.str.rstrip()
    climatology.columns = climatology.columns.str.replace('# ', '')
    climatology = climatology.drop_duplicates()
    return climatology


def import_holfuy(fpath=None, station=None, year=None, ice_only=True, exclude_keywords=True, format=None):
    """

    Parameters
    ----------
    fpath
    ice_only
    exclude_keywords
    format: standard: original format from johannes

    Returns
    -------

    """
    if not fpath:
        fpath = get_path_holfuy(station, format)
    if format=='standard_csv':
        cread = pd.read_csv(fpath, index_col=None, parse_dates=[0])

        # exclude some of the critical days
        if exclude_keywords:
            cread = cread[~cread.key_remarks.str.contains("SETUP", na=False)]
            # cread = cread[~cread.key_remarks.str.contains("REDRILL", na=False)]
            cread = cread[~cread.key_remarks.str.contains("TEARDOWN", na=False)]

        if ice_only:
            cread = cread[cread.phase == 'i']  # exclude snow

        return cread

    elif format=='standard_point':
        head = pd.read_csv(
            fpath, sep=';', skiprows=1, skipinitialspace=True,
            header=0, nrows=0, encoding='latin1').columns
        usecols = np.arange(len(head))
        colnames = head
        df = pd.read_csv(fpath,
                         skiprows=4, sep=' ', skipinitialspace=True,
                         usecols=usecols, header=None,
                         names=colnames, dtype={'date0': str,
                                                'date1': str})
        df.columns = df.columns.str.rstrip()
        df.columns = df.columns.str.replace('# ', '')
        df = df.drop_duplicates()
        # manual remove error in the glazioarch files
        try:
            if df.loc[269]['name']=='HF1008':
                df = df.drop(index=269)
        except:
            pass

        if not station:
            df = df[df['name'].str.contains('|'.join('HF'))]
        else:
            df = df[df['name'].str.contains('HF'+str(station))]
        selection = df[['date0', 'date1', 'mb_we']]

        date0 = [dt.datetime(int(x[0:4]), int(x[4:6]), int(x[6:])) for x in selection['date0'].values]
        date1 = [dt.datetime(int(x[0:4]), int(x[4:6]), int(x[6:])) for x in selection['date1'].values]
        selection = selection.assign(date0=date0)
        selection = selection.assign(date1=date1)

        selection = selection[['date0', 'date1', 'mb_we']]
        if year is not None:
            selection = selection[selection['date0'].dt.year == year]
        return selection

    elif format=='csv_2021':
        cread = pd.read_csv(fpath, index_col=None, parse_dates=[0], dayfirst=True)
        return cread
    elif format=='winter':
        cread = pd.read_excel(fpath, index_col=None)#, parse_dates=[0], dayfirst=True)
        return cread
    elif format=='mh_2021':
        #path_mh = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\holfuy_manual_readings_2021\\matthias\\fin1001_p700_holfuy2021.xlsx'
        holfuy_mh = pd.read_excel(fpath, index_col=None, skiprows=8, usecols=[0, 6], names=['date', 'mb[m.w.e]'])
        holfuy_mh = holfuy_mh.loc[holfuy_mh['date'].notnull()]
        holfuy_mh['mb[m.w.e]'] = holfuy_mh['mb[m.w.e]']/100
        return holfuy_mh
    else:
        return None


def plot_clim(df, year=None, label=None):
    """

    Parameters
    ----------
    df: dataframe containing the data to be plotted (pd DataFrame)
    t_init: initial time (np.datetime)
    t_end: final time (np.datetime)

    Returns
    -------
    plot
    """
    #fig, ax = plt.subplots(figsize=(10, 5))
    xtime = pd.date_range(
        dt.datetime(year - 1, 10, 1), periods=366)  # CHANGEEEEE

    if ((xtime.month == 2) & (xtime.day == 29)).any():
        xvals = np.arange(366)
    else:
        xtime = xtime[:-1]
        xvals = np.arange(365)
    try:
        std = df.groupby(['DOY']).std().reset_index()
        df = df.groupby(['DOY']).mean().reset_index()
    except:
        print('No column DOY')
    if '1001' in label:
        color = 'tab:blue'
    elif '1008' in label:
        color = 'tab:orange'
    elif '1002' in label:
        color='tab:green'
    elif '1006' in label:
        color='tab:blue'
    elif '1007' in label:
        color='tab:orange'
    elif '1003' in label:
        color='tab:blue'
   # try:
        #ax.plot(xtime, np.roll((df['balance(b)'].values / 1000), -274)[:-1], label=label, color=color)
        #ax.fill_between(xtime, np.roll(((df['balance(b)'].values - std['balance(b)'].values)/ 1000), -274)[:-1],
         #               np.roll(((df['balance(b)'].values + std['balance(b)'].values)/ 1000), -274)[:-1], color=color, alpha=0.1)
    #except:
        #ax.plot(xtime, np.roll((df['balance(b)'].values / 1000), -275)[:-2], label=label, color=color)
        #ax.fill_between(xtime, np.roll(((df['balance(b)'].values - std['balance(b)'].values)/ 1000), -274)[:-2],
                       # np.roll(((df['balance(b)'].values + std['balance(b)'].values)/ 1000), -274)[:-2], color=color, alpha=0.1)


def plot_holfuy(df, format=None, label=None):
    """

    Parameters
    ----------
    df: dataframe containing the data to be plotted (pd DataFrame)
    t_init: initial time (np.datetime)
    t_end: final time (np.datetime)

    Returns
    -------
    plot
    """
    #fig, ax = plt.subplots(figsize=(10, 5))

    xtime = df['date'].values
    #xtime = [pd.to_datetime(x, format="%d/%m/%y") for x in xtime]
    rounded = xtime.astype('datetime64[s]')
    xtime = [x.astype(dt.datetime) for x in rounded]
    xtime_doy = [x.timetuple().tm_yday for x in xtime]
    if format=='standard':
        dh = df['dh'].values
        dh_mwe = np.negative(dh) * rho_ice/rho_w
        cum_mb = np.nancumsum(dh_mwe)
    elif format=='csv_2021':
        cum_mb = df['mb[m.w.e]'].values
    if 'mh' in label:
        marker = 'x'
    else:
        marker='o'

    if '1001' in label:
        color = 'tab:purple'
        dy = 0
    elif '1008' in label:
        color = 'tab:orange'
        dy = 0
    elif '1002' in label:
        dy = 0
        color='tab:green'
    elif '1006' in label:
        color='tab:blue'
        dy = 0
    elif '1007' in label:
        dy = 0
        color='tab:orange'
    elif '1003' in label:
        dy = 0
        color='tab:blue'
    #ax.plot(xtime, cum_mb + dy, label=label, color=color)
    #ax.scatter(xtime, cum_mb + dy, marker='x', label=label, color='black', s= 10)


def plot_winter(df):
    xtime = df['final_date'].values
    cum_mb = df['mb[m.w.e]'].values
    rounded = xtime.astype('datetime64[s]')
    xtime = [x.astype(dt.datetime) for x in rounded]
    xtime_doy = [x.timetuple().tm_yday for x in xtime]
    #ax.scatter(xtime, cum_mb, s=10)


def plot_insitu():
    x = [dt.datetime(2021, 4, 8), dt.datetime(2021, 7, 30), dt.datetime(2021, 9, 22)]
    y = [0.4, np.negative(479-183)/100 +0.4, np.negative(479-183+(312-47))/100+0.4]
    #ax.scatter(x, y)


def plot_all(df_clim, df_holf):
    plot_clim(df_clim)
    plot_holfuy(df_holf)
    plt.show()


def interpolate_holfuy_df(df, mode=None):
    if mode is not None:
        time = np.arange(df['date'].values[0], df['date'].values[-1])
    else:
        time = np.arange(df['date'].values[0], df['date'].values[-1], dt.timedelta(days=1))
    not_yet_interpolated_holfuy = []
    for t in time:
        if t in df['date'].values:
            not_yet_interpolated_holfuy.append(df[df['date'] == t]['mb[m.w.e]'].values[0])
        else:
            not_yet_interpolated_holfuy.append(np.nan)

    interpolated_df = pd.DataFrame({'date': time, 'mb[m.w.e]': not_yet_interpolated_holfuy})
    if interpolated_df['mb[m.w.e]'].isnull().values.any():
        interpolated_df['mb[m.w.e]'] = interpolated_df['mb[m.w.e]'].interpolate(axis=0)
    return interpolated_df


def cut_clim_to_holfuy_period(clim_df, holfuy_df):
    first_date = np.min(holfuy_df['date'])
    last_date = np.max(holfuy_df['date'])
    indexes = clim_df[(clim_df['Month'] < first_date.month)
                        | (clim_df['Month'] == first_date.month) & (clim_df['Day'] < first_date.day)
                        | (clim_df['Month'] > last_date.month)
                        | (clim_df['Month'] == last_date.month) & (clim_df['Day'] > last_date.day)].index
    # droping mutiple rows based on column value
    clim_df = clim_df.drop(indexes)
    return clim_df


def get_path_holfuy(station, format=None):
    if format=='csv_2021':
        folder = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\final_readings_ablation\\'
        filename = str(station)+'_abl_sum.csv'
        path = os.path.join(folder, filename)
        return path

    elif format=='standard_point':
        folder = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2020\\holfuy_manual_readings_2020\\'
        if station==1001 or station==1008:
            filename = 'findelen_intermediate.dat'
        elif station==1002 or station==1006 or station==1007:
            filename = 'rhone_intermediate.dat'
        elif station==1003:
            filename = 'plainemorte_intermediate.dat'

        path = os.path.join(folder, filename)
        return path


def get_path_clim(station):
    if station == 1008:
        path = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\climatology\\point_mb_Aaron\\cumulative_fin_200.dat'
    elif station == 1001:
        path = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\climatology\\point_mb_Aaron\\cumulative_fin_700.dat'
    elif station == 1003:
        path = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\climatology\\point_mb_Aaron\\cumulative_plm_P6.dat'
    elif station == 1007:
        path = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\climatology\\point_mb_Aaron\\cumulative_rho_P5.dat'
    elif station == 1002:
        path = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\climatology\\point_mb_Aaron\\cumulative_rho_P6.dat'
    elif station == 1006:
        path = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\climatology\\point_mb_Aaron\\cumulative_rho_P8.dat'
    return path


def get_bias(station, year, visualize=True):
    # holfuy
    if year == 2021:
        holfuy = import_holfuy(station=station, format='csv_2021')
        xtime_h = holfuy['date'].values
        rounded = xtime_h.astype('datetime64[s]')
        xtime_h = [x.astype(dt.datetime) for x in rounded]
        cum_mb = holfuy['mb[m.w.e]'].values
        holfuy_new = pd.DataFrame({'date': xtime_h, 'mb[m.w.e]': cum_mb})
    else:
        holfuy = import_holfuy(station=station, year=year, format='standard_point')
        xtime_h = holfuy['date1'].values
        cum_mb = np.cumsum(holfuy['mb_we'].values / 1000)
        holfuy_new = pd.DataFrame({'date': xtime_h, 'mb[m.w.e]': cum_mb})
    # clim
    xtime_c = pd.date_range(dt.datetime(year - 1, 10, 1), periods=366)
    if ((xtime_c.month == 2) & (xtime_c.day == 29)).any():
        pass
    else:
        xtime_c = xtime_c[:-1]

    clim = import_climatology(get_path_clim(station))
    clim = clim[clim['Hyd.year'] == year]

    # interpolate holfuy data to have values every day
    interpolated_df = interpolate_holfuy_df(holfuy_new)
    mb_begin_season = get_mb_begin_period(interpolated_df, station, year)
    interpolated_df['mb[m.w.e]'] = interpolated_df['mb[m.w.e]'].values + mb_begin_season


    #if visualize:
        #fig, ax = plt.subplots(figsize=(10, 7))
        #ax.plot(interpolated_df['date'], interpolated_df['mb[m.w.e]'], label='2020')
        #ax.scatter(interpolated_df['date'], interpolated_df['mb[m.w.e]'], marker='x', color='black', s=10)
        #try:
        #    ax.plot(xtime_c, np.roll((clim['balance(b)'].values / 1000), -365), label=str(year))
        #except:
        #    ax.plot(xtime_c, np.roll((clim['balance(b)'].values / 1000), -366)[:-1], label=str(year))
        #plt.legend()
        #plt.title('Station : ' + str(station)+ ' year : ' + str(year))
        #plt.xlabel('Time')
        #plt.ylabel('Mass Balance [m.w.e]')
        #plt.show()

    # cut clim to match the length of holfuy data
    clim_cutted = cut_clim_to_holfuy_period(clim, interpolated_df)

    bias_y = interpolated_df['mb[m.w.e]'].values - clim_cutted['balance(b)'].values / 1000
    mean_bias_y = np.nanmean(bias_y)
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.text(0.25, 0.13, 'mean bias: ' + str(round(mean_bias_y, 3)) + ' m.w.e')
        ax.plot(interpolated_df['date'].values, interpolated_df['mb[m.w.e]'].values, label='holfuy', color='tab:blue')
        ax.scatter(interpolated_df['date'], interpolated_df['mb[m.w.e]'], marker='x', color='black', s=10)
        ax.plot(interpolated_df['date'].values, clim_cutted['balance(b)'].values / 1000, label='model', color='tab:orange')
        ax.plot(interpolated_df['date'].values, bias_y, label='bias', color='tab:red')
        ax.fill_between(interpolated_df['date'].values, bias_y, alpha=0.3, color='tab:red')
        plt.legend()
        plt.title('year ' + str(year) +', station ' + str(station))
        plt.suptitle('MODEL VS HOLFUY', fontweight="bold")
        plt.xlabel('Time')
        plt.ylabel('Mass Balance [m.w.e]')
        plt.show()
    return mean_bias_y


def get_bias_clim(station, year, visualize=True):
    if year == 2021:
        holfuy = import_holfuy(station=station, format='csv_2021')
        xtime_h = holfuy['date'].values
        rounded = xtime_h.astype('datetime64[s]')
        xtime_h = [x.astype(dt.datetime) for x in rounded]
        xtime_doy = [x.timetuple().tm_yday for x in xtime_h]  #
        cum_mb = holfuy['mb[m.w.e]'].values
        holfuy_new = pd.DataFrame({'date': xtime_doy, 'mb[m.w.e]': cum_mb})
    else:
        holfuy = import_holfuy(station=station, year=year, format='standard_point')
        xtime_h = holfuy['date1'].values
        rounded = xtime_h.astype('datetime64[s]')  #
        xtime_h = [x.astype(dt.datetime) for x in rounded]  #
        xtime_doy = [x.timetuple().tm_yday for x in xtime_h]  #
        cum_mb = np.cumsum(holfuy['mb_we'].values / 1000)
        holfuy_new = pd.DataFrame({'date': xtime_doy, 'mb[m.w.e]': cum_mb})
    # clim
    xtime_c = pd.date_range(dt.datetime(year - 1, 10, 1), periods=366)
    if ((xtime_c.month == 2) & (xtime_c.day == 29)).any():
        pass
    else:
        xtime_c = xtime_c[:-1]

    clim = import_climatology(get_path_clim(station))

    clim_std = clim.groupby(['DOY']).std().reset_index()  #
    clim = clim.groupby(['DOY']).mean().reset_index()  #
    interpolated_df = interpolate_holfuy_df(holfuy_new, mode='DOY')  #

    first_date = np.min(interpolated_df['date'])
    last_date = np.max(interpolated_df['date'])
    indexes = clim[(clim['DOY'] < first_date) | (clim['DOY'] > last_date)].index  #

    # droping mutiple rows based on column value
    clim_cutted = clim.drop(indexes)
    clim_std_cutted = clim_std.drop(indexes)

    bias_y = interpolated_df['mb[m.w.e]'].values - clim_cutted['balance(b)'].values / 1000

    if visualize:
        if bias_only:
            ax1.plot(clim_cutted['DOY'].values, bias_y,
                    label='bias' + 'Station : ' + str(station) + ' year : ' + str(year))
            ax1.fill_between(clim_cutted['DOY'].values, bias_y, alpha=0.2)
        else:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.plot(interpolated_df['date'].values, interpolated_df['mb[m.w.e]'].values, label='holfuy', color='tab:blue')
            ax.scatter(interpolated_df['date'].values, interpolated_df['mb[m.w.e]'].values, marker='x', color='black', s=10)
            ax.plot(clim_cutted['DOY'].values, clim_cutted['balance(b)'].values / 1000, label='climatology', color='tab:orange')
            ax.fill_between(clim_cutted['DOY'].values, (clim_cutted['balance(b)'].values - clim_std_cutted['balance(b)'].values)/ 1000 ,
                            (clim_cutted['balance(b)'].values + clim_std_cutted['balance(b)'].values)/ 1000 , color='tab:orange', alpha=0.1)

            ax.plot(clim_cutted['DOY'].values, bias_y, label='bias' + 'Station : ' + str(station)+ ' year : ' + str(year), color='tab:green')
            ax.fill_between(clim_cutted['DOY'].values, bias_y, color='tab:green', alpha=0.2)

            plt.title('year ' + str(year) + ', station ' + str(station))
            plt.suptitle('CLIMATOLOGY VS HOLFUY', fontweight="bold")
            plt.xlabel('Day of Year')
            plt.ylabel('Mass Balance [m.w.e]')
            plt.legend()
            plt.show()


def get_mb_begin_period(df, station, year):
    if year == 2021:
        return 0
    else:
        path = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\mb_begin_holfuy_period.xlsx'
        mb_begin_df = pd.read_excel(path)
        indexes = mb_begin_df[(mb_begin_df['station']!=station) | (mb_begin_df['date'].dt.year!=year)].index
        # droping mutiple rows based on column value
        mb_begin_df = mb_begin_df.drop(indexes)
        date = mb_begin_df['date'].values[0]
        mb = mb_begin_df['mb[m.w.e]'].values[0]
        print(date, mb)
        shift = mb - df[df['date']==date]['mb[m.w.e]'].values[0]
        return shift

if __name__ == '__main__':
    global fig1, ax1, rho_ice, rho_w, bias_only

    rho_ice = 900
    rho_w = 1000


    ### 2019 ###
    #path_holfuy = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2019\\holfuy_manual_readings_2019\\2019_manual_reading_1001.csv'
    path_holfuy198 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2019\\holfuy_manual_readings_2019\\2019_manual_reading_1008.csv'
    #path_holfuy2 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2019\\holfuy_manual_readings_2019\\2019_manual_reading_1003.csv'
    #path_holfuy = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2019\\holfuy_manual_readings_2019\\2019_manual_reading_1002.csv'
    #path_holfuy2 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2019\\holfuy_manual_readings_2019\\2019_manual_reading_1006.csv'
    #path_holfuy3 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2019\\holfuy_manual_readings_2019\\2019_manual_reading_1007.csv'
    #path_holfuy4 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2019\\holfuy_manual_readings_2019\\2019_manual_reading_1009.csv'

    ### 2020 ###
    path_holfuy201 = "C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2020\\holfuy_manual_readings_2020\\findelen_intermediate.dat"

    ### 2021 ###
    path_holfuy218 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\final_readings_ablation\\1008_abl_sum.csv'
    path_holfuy211 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\final_readings_ablation\\1001_abl_sum.csv'
    path_holfuy212 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\final_readings_ablation\\1002_abl_sum.csv'
    path_holfuy216 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\final_readings_ablation\\1006_abl_sum.csv'
    path_holfuy217 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\final_readings_ablation\\1007_abl_sum.csv'
    path_holfuy213 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\final_readings_ablation\\1004_abl_sum.csv'

    #readings mh
    path_holfuy_mh211 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\holfuy_manual_readings_2021\\matthias\\fin1001_p700_holfuy2021.xlsx'
    path_holfuy_mh218 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\holfuy_manual_readings_2021\\matthias\\fin1008_p200_holfuy2021.xlsx'
    path_holfuy_mh212 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\holfuy_manual_readings_2021\\matthias\\rho1002_p9_holfuy2021.xlsx'
    path_holfuy_mh216 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\holfuy_manual_readings_2021\\matthias\\rho1006_p9_holfuy2021.xlsx'
    path_holfuy_mh217 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\holfuy_manual_readings_2021\\matthias\\rho1007_p5_holfuy2021.xlsx'
    path_holfuy_mh213 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\holfuy_manual_readings_2021\\matthias\\plm1004_p6_holfuy2021.xlsx'

    # winter
    path_holfuy_w218 = 'C:\\Users\\Aaron\\Documents\\pubblications\\paper1_anomalies\\data\\holfuy\\2021\\holfuy_manual_readings_2021\\winter_reading\\1008_winter.xlsx'
    #holfuy_winter_1008 = import_holfuy(path_holfuy_w218, format='winter')
    # fin
    #clim_1008 = import_climatology(path_climf2)
    #clim_1001 = import_climatology(path_climf7)
    #holfuy_1001_21 = import_holfuy(path_holfuy211, format='csv_2021')
    #holfuy_1001_20 = import_holfuy(path_holfuy201, station=1001, year=2020, format='standard_point')
    #holfuy_1008 = import_holfuy(path_holfuy218, format='csv_2021')
    #holfuy_mh_1001 = import_holfuy(path_holfuy_mh211, format='mh_2021')
    #holfuy_mh_1008 = import_holfuy(path_holfuy_mh218, format='mh_2021')

    # rho
    #clim_1002 = import_climatology(path_climr6)
    #clim_1006 = import_climatology(path_climr8)
    #clim_1007 = import_climatology(path_climr5)
    #holfuy_1002 = import_holfuy(path_holfuy212, format='csv_2021')
    #holfuy_1006 = import_holfuy(path_holfuy216, format='csv_2021')
    #holfuy_1007 = import_holfuy(path_holfuy217, format='csv_2021')
    #holfuy_mh_1002 = import_holfuy(path_holfuy_mh212, format='mh_2021')
    #holfuy_mh_1006 = import_holfuy(path_holfuy_mh216, format='mh_2021')
    #holfuy_mh_1007 = import_holfuy(path_holfuy_mh217, format='mh_2021')


    # plm
    #clim_1003 = import_climatology(path_climp6)
    #holfuy_1003 = import_holfuy(path_holfuy213, format='csv_2021')
    #holfuy_mh_1003 = import_holfuy(path_holfuy_mh213, format='mh_2021')


    #fig, ax = plt.subplots(figsize=(7, 7))
    #plot_clim(clim_1001, label='1001', year=2021)
    #plot_clim(clim_1008, label='1008', year=2021)
    #plot_winter(holfuy_winter_1008)
    #plot_holfuy(holfuy_1001, label='1001 automatic',format='csv_2021')
    #plot_holfuy(holfuy_1008, label='1008 automatic', format='csv_2021')
    #plot_holfuy(holfuy_mh_1001, label='1001 mh', format='csv_2021')
    #plot_holfuy(holfuy_mh_1008,label='1008 mh', format='csv_2021')
    #plt.xlim([dt.datetime(2021, 7, 29), dt.datetime(2021, 9, 30)])
    #plt.legend()
    #plt.show()

    #fig, ax = plt.subplots(figsize=(7, 7))
    #plot_clim(clim_1002, label='1002', year=2021)
    #plot_clim(clim_1006, label='1006', year=2021)
    #plot_clim(clim_1007, label='1007', year=2021)
    #plot_holfuy(holfuy_1002, label='1002 automatic', format='csv_2021')
    #plot_holfuy(holfuy_1006, label='1006 automatic', format='csv_2021')
    #plot_holfuy(holfuy_1007, label='1007 automatic',format='csv_2021')
    #plot_holfuy(holfuy_mh_1002, label='1002 mh', format='csv_2021')
    #plot_holfuy(holfuy_mh_1006, label='1006 mh', format='csv_2021')
    #plot_holfuy(holfuy_mh_1007, label='1007 mh', format='csv_2021')
    #plt.xlim([dt.datetime(2021, 7, 29), dt.datetime(2021, 8, 30)])
    #plt.legend()
    #plt.show()

    #fig, ax = plt.subplots(figsize=(7, 7))
    #plot_clim(clim_1003, label='1003', year=2021)
    #plot_holfuy(holfuy_1003, label='1003 automatic', format='csv_2021')
    #plot_holfuy(holfuy_mh_1003, label='1003 mh',  format='csv_2021')
    #plt.ylim([-3.5, -1.8])
    #plt.xlim([dt.datetime(2021, 8, 10), dt.datetime(2021, 8, 30)])
    #plt.legend()
    #plt.show()


    ####
    bias_only= False
    years = [2021]
    # attention order of the paths resp stations must agre!!!
    #paths = [path_climf2]#, path_climf7, path_climr5, path_climr6, path_climr8, path_climp6]
    stations = [1008, 1001, 1007, 1002, 1006, 1003]
    bias = []

    if bias_only:
        fig1, ax1 = plt.subplots(figsize=(10, 7))

    for station in stations:
        for year in years:
            if bias_only:
                get_bias_clim(station, year)
            else:
                #get_bias_clim(station, year)
                bias_y = get_bias(station, year, visualize=True)
                bias.append(bias_y)
        print(bias, np.mean(bias))
    if bias_only:
        plt.xlabel('Day of Year')
        plt.ylabel('Mass Balance [m.w.e]')
        plt.legend()
        plt.show()

