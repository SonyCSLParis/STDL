import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_distances

# -------------------------
# STDC class definition
# -------------------------
class STDC:

    # -------------------------
    # initialization
    # -------------------------

    def __init__(self, raw_data=None, field_names=['id1', 'id2', 'timestamp'], timeframe='Y', time_type='actual'):

        # generate random data if no data is passed
        if raw_data is None:
            self.raw_data = self.random_data_gen()
        else:
            self.raw_data = raw_data
        
        # check for correct field input
        if not isinstance(field_names, list) or len(field_names) < 3:
            raise ValueError("field_names must be a list with three ordered elements: [leader_col, follower_col, timestamp_col]")
        
        self.field_names = field_names
        self.leader_col = field_names[0]
        self.follower_col = field_names[1]
        self.timestamp_col = field_names[2]
        self.timeframe = timeframe
        self.time_type = time_type

    # -------------------------
    # methods
    # -------------------------

    def data(self):
        return self.raw_data

    def view(self, n=5):
        return self.raw_data.head(n)

    def get_unique_entities(self, raw_data = None, leader_col=None):
        # assumes raw_data is None
        if raw_data is None:
            raw_data = self.raw_data

        # assumes leader_col is not provided
        if leader_col is None:
            leader_col = self.leader_col

        return raw_data[leader_col].unique()

    def get_unique_timeframes(self, raw_data=None):

        # assumes raw_data is None
        if raw_data is None:
            raw_data = self.raw_data

        if isinstance(raw_data.index, pd.MultiIndex):
            return raw_data.index.get_level_values(1).unique()
        elif 'timeframe' in raw_data.columns:
            return raw_data['timeframe'].unique()
        else:
            raise ValueError("DataFrame must have either a MultiIndex with a timeframe level or a 'timeframe' column.")

    def bin_timestamps(self, raw_data, timestamp_col, timeframe):
        raw_data['timeframe'] = raw_data[timestamp_col].dt.to_period(timeframe).apply(
            lambda r: str(r.start_time) + " to " + str(r.end_time)
        )
        return raw_data

    def get_timeframe_data(self, raw_data, timeframe_value, timeframe_col='timeframe'):
        return raw_data[raw_data[timeframe_col] == timeframe_value]

    def compute_biadjacency_matrix(self, raw_data=None, leader_col=None, follower_col=None, agg_func=None):

        # assumes raw_data is None
        if raw_data is None:
            raw_data = self.raw_data

        # Set default columns if not provided
        if leader_col is None:
            leader_col = self.leader_col
        if follower_col is None:
            follower_col = self.follower_col

        # needs binned df
        binned_df = self.bin_timestamps(raw_data, timestamp_col=self.timestamp_col, timeframe=self.timeframe)

        if agg_func is None:
            agg_func = pd.NamedAgg(column=follower_col, aggfunc='size')
            grouped = binned_df.groupby([leader_col, follower_col, 'timeframe']).size().reset_index(name='counts')
        else:  # attempt to add another aggregation function
            grouped = binned_df.groupby([leader_col, follower_col, 'timeframe']).agg(counts=(follower_col, agg_func)).reset_index()

        biadjacency_matrix = grouped.pivot(
            index=[leader_col, 'timeframe'], columns=follower_col, values='counts').fillna(0)

        return biadjacency_matrix

    def compute_relative_matrix(self, biadjacency_matrix, distance_function):
        tmp_index_tf = []
        relative_cosine_dist_matrices = pd.DataFrame()
        for tf in biadjacency_matrix.index.levels[1].unique():
            tmp_raw_data = biadjacency_matrix.xs(tf, level='timeframe')
            tmp_cosine_dist_matrix = pd.DataFrame(
                distance_function(tmp_raw_data),
                index=tmp_raw_data.index,
                columns=tmp_raw_data.index
            )
            relative_cosine_dist_matrices = pd.concat([relative_cosine_dist_matrices, tmp_cosine_dist_matrix], axis=0)
            tmp_index_tf.extend([tf] * len(tmp_cosine_dist_matrix))
        relative_cosine_dist_matrices['timeframe'] = tmp_index_tf
        relative_cosine_dist_matrices.set_index(['timeframe'], append=True, inplace=True)
        return relative_cosine_dist_matrices

    def get_relative_matrix(self, raw_data=None, agg_func = None, distance_function=cosine_distances, dimensions = None):
        
        # Assign default values if arguments are None
        if raw_data is None:
            raw_data = self.raw_data

        #if dimensions > 2:
            #pass

        #raw_data_binned = self.bin_timestamps(raw_data, timestamp_col=timestamp_col, timeframe=timeframe)
        biadjacency_matrix = self.compute_biadjacency_matrix(raw_data, leader_col=self.leader_col, follower_col=self.follower_col, agg_func=agg_func)
        relative_matrix = self.compute_relative_matrix(biadjacency_matrix, distance_function=distance_function)
        return relative_matrix

    def get_absolute_matrix(self, raw_data=None, agg_func = None, distance_function=cosine_distances, dimensions = None):
        
        # Assign default values if arguments are None
        if raw_data is None:
            raw_data = self.raw_data

        #if dimensions > 2:
            #pass

        #raw_data_binned = self.bin_timestamps(raw_data, timestamp_col=timestamp_col, timeframe=timeframe)
        biadjacency_matrix = self.compute_biadjacency_matrix(raw_data, leader_col=self.leader_col, follower_col=self.follower_col, agg_func=agg_func)
        absolute_matrix = pd.DataFrame(
            distance_function(biadjacency_matrix),
            index=biadjacency_matrix.index,
            columns=biadjacency_matrix.index
        )
        return absolute_matrix
    
    def calculate_positions(self, raw_data=None, agg_func=None, comparison='relative', distance_function=cosine_distances, dimensions=None):
        # assume no data was given
        if raw_data is None:
            raw_data = self.raw_data

        # get biadjacency matrix
        biadjacency_matrix = self.compute_biadjacency_matrix(raw_data, leader_col=self.leader_col, follower_col=self.follower_col, agg_func=agg_func)

        # get absolute matrix
        if comparison != 'relative':
            absolute_matrix = pd.DataFrame(distance_function(biadjacency_matrix), index=biadjacency_matrix.index, columns=biadjacency_matrix.index)
            return absolute_matrix

        # get relative matrix
        else:
            relative_matrix = self.compute_relative_matrix(biadjacency_matrix, distance_function=distance_function)
            return relative_matrix

    def random_data_gen(self, num_rows=1000, n_layer_1=3, n_layer_2=20,
                        start_dt=datetime(2020, 1, 1), end_dt=datetime(2025, 1, 1)):
        id1 = ["L1_" + str(x) for x in np.random.randint(n_layer_1, size=num_rows)]
        id2 = ["L2_" + str(x) for x in np.random.randint(n_layer_2, size=num_rows)]
        start_u = start_dt.timestamp()
        end_u = end_dt.timestamp()
        random_ts = np.random.uniform(start_u, end_u, num_rows)
        dt = sorted([datetime.fromtimestamp(ts) for ts in random_ts])
        raw_data = pd.DataFrame({
            'id1': id1,
            'id2': id2,
            'timestamp': dt
        })
        return raw_data