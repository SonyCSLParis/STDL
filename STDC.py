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
        
        self.biadjacency_matrix = biadjacency_matrix

        return biadjacency_matrix

    def compute_relative_matrix(self, biadjacency_matrix = None, agg_func = None, distance_function=cosine_distances):

        # if the user has not computed biadjacency matrix before or did not input one, calculate it
        if hasattr(self, 'biadjacency_matrix') and self.biadjacency_matrix is not None:
            biadjacency_matrix = self.biadjacency_matrix

        else:
            # get biadjacency matrix
            biadjacency_matrix = self.compute_biadjacency_matrix(self.raw_data, leader_col=self.leader_col, follower_col=self.follower_col, agg_func=agg_func)

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

        if hasattr(self, 'biadjacency_matrix') and self.biadjacency_matrix is not None:
            biadjacency_matrix = self.biadjacency_matrix

        else:
            # get biadjacency matrix
            biadjacency_matrix = self.compute_biadjacency_matrix(raw_data, leader_col=self.leader_col, follower_col=self.follower_col, agg_func=agg_func)

        relative_matrix = self.compute_relative_matrix(biadjacency_matrix, distance_function=distance_function)
        self.relative_matrix = relative_matrix # store it in the object
        return relative_matrix
    
    # def compute_absolute_matrix(self, biadjacency_matrix = None, distance_function=cosine_distances):

    #     absolute_matrix = pd.DataFrame(distance_function(biadjacency_matrix), index=biadjacency_matrix.index, columns=biadjacency_matrix.index)
    #     self.absolute_matrix = absolute_matrix # store it in the object

    #     return absolute_matrix

    def get_absolute_matrix(self, raw_data=None, agg_func = None, distance_function=cosine_distances, dimensions = None):
        
        # Assign default values if arguments are None
        if raw_data is None:
            raw_data = self.raw_data

        #if dimensions > 2:
            #pass

        #raw_data_binned = self.bin_timestamps(raw_data, timestamp_col=timestamp_col, timeframe=timeframe)

        if hasattr(self, 'biadjacency_matrix') and self.biadjacency_matrix is not None:
            biadjacency_matrix = self.biadjacency_matrix

        else:
            # get biadjacency matrix
            biadjacency_matrix = self.compute_biadjacency_matrix(raw_data, leader_col=self.leader_col, follower_col=self.follower_col, agg_func=agg_func)

        absolute_matrix = pd.DataFrame(
            distance_function(biadjacency_matrix),
            index=biadjacency_matrix.index,
            columns=biadjacency_matrix.index
        )
        self.absolute_matrix = absolute_matrix # store it in the object
        return absolute_matrix
    
    def calculate_positions(self, raw_data=None, agg_func=None, comparison='relative', distance_function=cosine_distances, dimensions=None):
        # assume no data was given
        if raw_data is None:
            raw_data = self.raw_data

        #if dimensions > 2:
            #pass

        if hasattr(self, 'biadjacency_matrix') and self.biadjacency_matrix is not None:
            biadjacency_matrix = self.biadjacency_matrix

        else:
            # get biadjacency matrix
            biadjacency_matrix = self.compute_biadjacency_matrix(raw_data, leader_col=self.leader_col, follower_col=self.follower_col, agg_func=agg_func)

        # get absolute matrix
        if comparison != 'relative':
            absolute_matrix = pd.DataFrame(distance_function(biadjacency_matrix), index=biadjacency_matrix.index, columns=biadjacency_matrix.index)
            self.absolute_matrix = absolute_matrix # store it in the object
            return absolute_matrix

        # get relative matrix
        else:
            relative_matrix = self.compute_relative_matrix(biadjacency_matrix, distance_function=distance_function)
            self.relative_matrix = relative_matrix # store it in the object
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
    
    def view_biadjacency_matrix(self):
        if not hasattr(self, 'biadjacency_matrix') or self.biadjacency_matrix is None:
            raise AttributeError("Biadjacency matrix has not been computed yet. Please compute it first.")
        return self.biadjacency_matrix

    
    def calculate_velocities(self, raw_data=None, agg_func=None, comparison='relative', distance_function=cosine_distances, dimensions=None):

        """
        Calculates the velocities (cosine distances) between consecutive timeframes.

        Parameters:
        - raw_data (pd.DataFrame): Optional raw data. Defaults to self.raw_data.
        - agg_func: Aggregation function for computing the biadjacency matrix. Defaults to counts.
        - distance_function: Distance metric function (default: cosine_distances).

        Returns:
        - velocities_df (pd.DataFrame): DataFrame with columns:
            ['Node', 'From', 'To', 'Velocity']
        """

        # if comparison is 'relative':

        #     timeframes = 


        # elif comparison is 'absolute':

        #     timeframes = 

        # else:

        # assume user doesn't pass any data
        if raw_data is None:
            raw_data = self.raw_data

        if hasattr(self, 'biadjacency_matrix') and self.biadjacency_matrix is not None:
            biadjacency_matrix = self.biadjacency_matrix

        else:
            # Step 1: Build biadjacency matrix (with timeframes included)
            biadjacency_matrix = self.compute_biadjacency_matrix(raw_data, leader_col=self.leader_col, follower_col=self.follower_col, agg_func=agg_func)

        # Extract unique timeframes (sorted)
        timeframes = biadjacency_matrix.index.get_level_values("timeframe").unique().sort_values()

        
        velocity_records = []

        # Step 2: Loop over consecutive timeframes
        for i in range(len(timeframes) - 1):
            current_tf = timeframes[i]
            next_tf = timeframes[i + 1]

            # Get data for the two timeframes
            current_data = biadjacency_matrix.xs(current_tf, level="timeframe")
            next_data = biadjacency_matrix.xs(next_tf, level="timeframe")

            # Align indices (leaders)
            all_indices = current_data.index.union(next_data.index)
            current_data = current_data.reindex(all_indices).fillna(0)
            next_data = next_data.reindex(all_indices).fillna(0)

            # Step 3: Compute velocities node-wise
            for node in all_indices:
                vec1 = current_data.loc[node].values
                vec2 = next_data.loc[node].values

                if np.all(vec1 == 0) or np.all(vec2 == 0):
                    velocity = np.nan
                else:
                    velocity = distance_function([vec1], [vec2])[0][0]

                velocity_records.append({
                    "Node": node,
                    "t1": current_tf,
                    "t2": next_tf,
                    "Velocity": velocity
                })

        # Step 4: Convert to DataFrame
        velocities_df = pd.DataFrame(velocity_records)

        return velocities_df