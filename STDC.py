import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA

# -------------------------
# STDC definition
# -------------------------
class STDC:

    def __init__(self, 
                 raw_data=None, 
                 field_names=['id1', 'id2', 'timestamp'],
                 timeframe='Y', 
                 time_type='actual',
                 agg_func=None,
                 distance_function=cosine_distances,
                 dimensions=None,
                 comparison='relative',

                 # confige of random_data_gen()
                 num_rows=1000,
                 n_layer_1=3,
                 n_layer_2=20,
                 start_dt=datetime(2020, 1, 1),
                 end_dt=datetime(2025, 1, 1)):

        # check for correct field input
        if not isinstance(field_names, list) or len(field_names) < 3 or type(field_names) is not list:
            raise ValueError("field_names must be a list with 3 ordered elements: [projected_layer, other_layer, timestamp_col]; default: ['id1', 'id2', 'timestamp']")
        
        ### SHOULD ALL VARIABLE BE PRIVATE???
        
        # set up column names
        self.projected_layer = field_names[0]
        self.other_layer = field_names[1]
        self.timestamp_col = field_names[2]

        self.biadjacency_matrix = None
        self.reduced_positions = None
        self.positions = None
        self.velocities = None

        # global defaults
        self.__timeframe = timeframe
        self.__time_type = time_type
        self.__agg_func = agg_func
        self.__distance_function = distance_function
        self.__dimensions = dimensions
        self.__comparison = comparison

        # random data config
        self.__num_rows = num_rows
        self.__nlayer1 = n_layer_1
        self.__nlayer2 = n_layer_2
        self.__start_dt = start_dt
        self.__end_dt = end_dt

        # generate random data if no data is passed
        self.raw_data = raw_data if raw_data is not None else self.random_data_gen()

    # --------------------------------
    # analysis & preprocessing methods
    # --------------------------------

    def random_data_gen(self):
        """Generate random sample bipartite data using self defaults."""
        id1 = ["L1_" + str(x) for x in np.random.randint(self.__nlayer1, size=self.__num_rows)]
        id2 = ["L2_" + str(x) for x in np.random.randint(self.__nlayer2, size=self.__num_rows)]
        start_u = self.__start_dt.timestamp()
        end_u = self.__end_dt.timestamp()
        random_ts = np.random.uniform(start_u, end_u, self.__num_rows)
        dt = sorted([datetime.fromtimestamp(ts) for ts in random_ts])
        raw_data = pd.DataFrame({
            'id1': id1,
            'id2': id2,
            'timestamp': dt
        })
        return raw_data
    
    def calculate_timeframe(self):
        """Add a timeframe column by binning timestamps."""
        if self.__time_type == 'actual':
            #EXAMPLE: self.__timeframe = "%Y" for yearly bins
            #EXAMPLE: self.__timeframe = "%Y-%m" for monthly bins
            #EXAMPLE: self.__timeframe = "%Y-%W" for weekly bins
            #EXAMPLE: self.__timeframe = "%Y-%m-%d" for daily bins
            if not isinstance(self.__timeframe, str):
                raise ValueError("If time_type is actual, timeframe must be a string.")
            self.raw_data['timeframe'] =  self.raw_data['timestamp'].dt.strftime(self.__timeframe)
        elif self.__time_type == 'intrinsic':
            if not isinstance(self.__timeframe, int):
                raise ValueError("If time_type is intrinsic, timeframe must be an integer.")
            self.raw_data.sort_values(by=[self.timestamp_col], inplace=True)
            self.raw_data['timeframe'] = range(len(self.raw_data))
            self.raw_data['timeframe'] = np.floor(self.raw_data['timeframe'] / self.__timeframe).astype(int)
        else:
            raise NotImplementedError("Not implemented yet.")
        return self.raw_data    

    def calculate_biadjacency_matrix(self):
        """Builds biadjacency matrix (leaders x followers per timeframe)."""

        # check if timeframe was calculated
        if 'timeframe' not in self.raw_data.columns:
            self.calculate_timeframe()

        if self.__agg_func is None:
            grouped = self.raw_data.groupby([self.projected_layer, self.other_layer, 'timeframe']).size().reset_index(name='counts')
        else:
            raise NotImplementedError("Other aggregation functions have not been implemented yet.")

        self.biadjacency_matrix = grouped.pivot(
            index=[self.projected_layer, 'timeframe'], columns=self.other_layer, values='counts'
        ).fillna(0)
        
        return self.biadjacency_matrix

    def calculate_positions(self):
        """calculates relative (per __timeframe) distance matrices or absolute matrix (all timeframes combined)."""

        #check if biadjacency_matrix exists, if not calculate it
        if self.biadjacency_matrix is None:
            self.calculate_biadjacency_matrix()

        if self.__comparison == 'relative':
            relative_cosine_dist_matrices = pd.DataFrame()

            for tf in self.biadjacency_matrix.index.levels[1].unique():
                tmp_raw_data = self.biadjacency_matrix.xs(tf, level='timeframe')
                multi_index = pd.MultiIndex.from_arrays([tmp_raw_data.index, [tf]*len(tmp_raw_data.index)], names=[self.projected_layer, 'timeframe'])
                tmp_cosine_dist_matrix = pd.DataFrame(
                    self.__distance_function(tmp_raw_data),
                    index=multi_index,
                    columns=tmp_raw_data.index
                )
                relative_cosine_dist_matrices = pd.concat([relative_cosine_dist_matrices, tmp_cosine_dist_matrix], axis=0)

            self.positions = relative_cosine_dist_matrices
            return self.positions
        
        else:
            #store absolute_positions as an object attribute
            self.positions = pd.DataFrame(
                self.__distance_function(self.biadjacency_matrix),
                index=self.biadjacency_matrix.index,
                columns=self.biadjacency_matrix.index
            )
            return self.positions

    
    # add dim reduction possibility here
    def calculate_reduced_positions(self):
        """Reduce the dimensionality of the positions if needed."""

        if self.positions is None:
            self.calculate_positions()

        if self.__dimensions is not None:
            # take the POSITIONS and reduce dimensionality using e.g. PCA with 2 components
            pca = PCA(n_components=self.__dimensions)
            self.reduced_positions = pd.DataFrame(pca.fit_transform(self.positions), index=self.positions.index)
            # OUTPUT SOMEWHERE: explained variance ratio: pca.explained_variance_ratio_
            return self.reduced_positions
        else:
            self.reduced_positions = self.positions
            return self.reduced_positions

    def calculate_velocities(self):
        """Calculates the velocities of the entities as the difference of the position between consecutive timeframes."""
        
        if self.reduced_positions is None:
            self.reduced_positions = self.calculate_reduced_positions()

        # assuming reduced_positions has a MultiIndex with levels: (node, timeframe)
        velocities = pd.DataFrame()

        for node in self.reduced_positions.index.get_level_values(self.projected_layer).unique():
            tmp = self.reduced_positions.xs(node, level=self.projected_layer).sort_index()
            tmp2 = tmp.diff().dropna()
            tmp2.index = pd.MultiIndex.from_arrays([[node]*len(tmp2), tmp.index[:-1], tmp.index[1:]], names=[self.projected_layer, 't1', 't2'])
            velocities = pd.concat([velocities, tmp2], axis=0)

        self.velocities = velocities
        return self.velocities

    def calculate_custom_distance_velocities(self):
        """
        Calculates the velocities of the entities as the difference of the position between consecutive timeframes.
        Returns:
        - velocities_df (pd.DataFrame): DataFrame with columns: ['Node', 't1', 't2', 'Velocity']
        """
        # Check if reduced_positions already exist
        if self.reduced_positions is None:
            self.reduced_positions = self.calculate_reduced_positions()

        # 5. Extract unique timeframes (sorted)
        if self.positions.index.nlevels == 2:
            # MultiIndex: (node, timeframe)
            timeframes = self.positions.index.get_level_values("timeframe").unique().sort_values()
        elif self.positions.index.nlevels == 1 and 'timeframe' in self.positions.columns:
            # Single index, timeframe as column
            timeframes = self.positions['timeframe'].unique().sort()
        else:
            raise ValueError("Positions index must contain 'timeframe'.")
        
        velocity_records = []

        # 6. Loop over consecutive timeframes
        for i in range(len(timeframes) - 1):
            current_tf = timeframes[i]
            next_tf = timeframes[i + 1]

            try:
                current_data = self.positions.xs(current_tf, level="timeframe")
                next_data = self.positions.xs(next_tf, level="timeframe")
            except Exception:
                # fallback for single index
                current_data = self.positions[self.positions['timeframe'] == current_tf].drop(columns='timeframe')
                next_data = self.positions[self.positions['timeframe'] == next_tf].drop(columns='timeframe')

            all_indices = current_data.index.union(next_data.index)
            current_data = current_data.reindex(all_indices).fillna(0)
            next_data = next_data.reindex(all_indices).fillna(0)

            for node in all_indices:
                vec1 = current_data.loc[node].values
                vec2 = next_data.loc[node].values

                if np.all(vec1 == 0) or np.all(vec2 == 0):
                    velocity = np.nan
                else:
                    velocity = self.__distance_function([vec1], [vec2])[0][0]

                velocity_records.append({
                    "Node": node,
                    "t1": current_tf,
                    "t2": next_tf,
                    "Velocity": velocity
                })

        self.cd_velocities = pd.DataFrame(velocity_records)

        return self.cd_velocities

    # --------------------------------
    # statistics methods
    # --------------------------------
  
    # --------------------------------
    # visualisation methods
    # --------------------------------

    def plot_pca_interactive():
        pass

    def plot_combined_graph():
        pass

    def plot_kde_overlay():
        pass