import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def get_unique_entities(df,layer1):
    return df[layer1].unique()

def get_unique_timeframes(df):
    # If MultiIndex, get level 1 (usually timeframe), else get unique values from 'timeframe' column
    if isinstance(df.index, pd.MultiIndex):
        return df.index.get_level_values(1).unique()
    elif 'timeframe' in df.columns:
        return df['timeframe'].unique()
    else:
        raise ValueError("DataFrame must have either a MultiIndex with a timeframe level or a 'timeframe' column.")

def bin_timestamps(df, timestamp_col, timeframe):

    '''
    adds a new column called ‘timeframe‘ and bins the timestamps user specified timeframes
    '''

    df['timeframe'] = df[timestamp_col].dt.to_period(timeframe).apply(lambda r: str(r.start_time)+" to " + str(r.end_time))

    return df

def get_timeframe_data(df, timeframe_value, timeframe_col='timeframe'):
    """
    Returns rows from df where the timeframe_col matches the specified timeframe_value.
    Example timeframe_value: '2020-01-01 00:00:00 to 2021-01-01 00:00:00'
    """
    return df[df[timeframe_col] == timeframe_value]

def compute_biadjacency_matrix(df, layer1, layer2):
    """
    Groups and pivots the dataframe by id1, id2, and a timeframe period.
    Args:
        df: Input DataFrame (e.g., from random_data_gen)
        layer1: Name of the first ID column
        layer2: Name of the second ID column
    Returns:
        biadjacency_matrix: Pivoted DataFrame with id2 values as columns and counts as values
    """
    grouped_df = df.groupby([layer1, layer2, 'timeframe']).size().reset_index(name='counts')
    biadjacency_matrix = grouped_df.pivot(index=[layer1, 'timeframe'], columns=layer2, values='counts').fillna(0)
    
    return biadjacency_matrix

def compute_relative_matrix(biadjacency_matrix, distance_function):

    tmp_index_tf = []

    relative_cosine_dist_matrices = pd.DataFrame()

    for tf in biadjacency_matrix.index.levels[1].unique():
        tmp_df = biadjacency_matrix.xs(tf, level='timeframe') #subset matrix
        tmp_cosine_dist_matrix = pd.DataFrame(distance_function(tmp_df), index=tmp_df.index, columns=tmp_df.index)
        relative_cosine_dist_matrices = pd.concat([relative_cosine_dist_matrices, tmp_cosine_dist_matrix], axis=0)
        tmp_index_tf.extend([tf] * len(tmp_cosine_dist_matrix))

    relative_cosine_dist_matrices['timeframe'] = tmp_index_tf
    relative_cosine_dist_matrices.set_index(['timeframe'], append=True, inplace=True)

    return relative_cosine_dist_matrices

def get_relative_matrix(df, layer1 = 'id1', layer2 = 'id2', timestamp_col = 'timestamp', timeframe = 'Y', distance_function = cosine_distances):

    df_binned = bin_timestamps(df, timestamp_col = timestamp_col, timeframe = timeframe)
    biadjacency_matrix = compute_biadjacency_matrix(df_binned, layer1=layer1, layer2 = layer2)
    relative_matrix = compute_relative_matrix(biadjacency_matrix, distance_function = distance_function)

    return relative_matrix

def get_absolute_matrix(df, layer1 = 'id1', layer2 = 'id2', timestamp_col = 'timestamp', timeframe = 'Y', distance_function = cosine_distances):

    df_binned = bin_timestamps(df, timestamp_col = timestamp_col, timeframe = timeframe)

    biadjacency_matrix = compute_biadjacency_matrix(df_binned, layer1 = layer1, layer2 = layer2)

    absolute_matrix = pd.DataFrame(distance_function(biadjacency_matrix), index=biadjacency_matrix.index, columns=biadjacency_matrix.index)

    return absolute_matrix