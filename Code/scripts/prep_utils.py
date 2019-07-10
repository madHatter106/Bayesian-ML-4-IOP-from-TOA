import numpy as np


def make_features_from_dt(df, cleanup=True):
    yeardays = max(df.datetime.dt.dayofyear.max(), 365)
    dayminutes = 24 * 60
    doy_arg = 2 * np.pi * df.datetime.dt.dayofyear / yeardays
    mod_arg = 2 * np.pi * (df.datetime.dt.hour * 60 + df.datetime.dt.minute) / dayminutes
    df['sin_doy'] = np.sin(doy_arg)
    df['cos_doy'] = np.cos(doy_arg)
    df['sin_minofday'] = np.sin(mod_arg)
    df['cos_minofday'] = np.cos(mod_arg)
    if cleanup:
        df.drop('datetime', axis=1, inplace=True)
    
    
def make_features_from_latlon(df, cleanup=True):
    df['x'] = np.cos(np.deg2rad(df.lat)) * np.cos(np.deg2rad(df.lon))
    df['y'] = np.cos(np.deg2rad(df.lat)) * np.sin(np.deg2rad(df.lon))
    df['z'] = np.sin(np.deg2rad(df.lat))
    if cleanup:
        df.drop(['lat', 'lon'], axis=1, inplace=True)

def log_transform_feature(df, feature, offset=1e-6, cleanup=True):
    try:
        df['log10_' + feature] = np.log10(df[feature]+offset)
    except TypeError:
        for feature_i in feature:
            df['log10_' + feature_i] = np.log10(df[feature_i]+offset)
    if cleanup:
        df.drop(feature, axis=1, inplace=True)