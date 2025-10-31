import pandas as pd
import numpy as np
from datetime import datetime
import scipy.sparse as sp

#plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

#distance functions
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances

#dimensionality reduction
from sklearn.decomposition import PCA
import umap

#network libraries
import graph_tool.all as gt
import igraph as ig

# -------------------------
# STDC definition
# -------------------------
class STDC:

    def __init__(self, 
                 raw_data=None, 
                 field_names=['id1', 'id2', 'timestamp'],
                 timeframe='%Y',
                 time_type='actual',
                 agg_func=None,
                 distance_function=None, #expects a string
                 dimensions=None,
                 reduction_function = None,
                 community_detection = "Leiden",
                 comparison='relative',

                 # configuration of random_data_gen()
                 num_rows=1000,
                 n_layer_1=3,
                 n_layer_2=20,
                 start_dt=datetime(2020, 1, 1),
                 end_dt=datetime(2025, 1, 1)):
        
        """
        Initialize an STDC (Social Thermodynamics Class) object.

        This constructor sets up the framework for analyzing temporal bipartite networks, 
        where nodes from one layer are projected against another over time. 
        If no dataset is provided, a synthetic dataset is generated using 
        'random_data_gen()'.

        Parameters
        ----------
        raw_data : pd.DataFrame, optional
            Input dataset containing bipartite edges with timestamps. Must have at least 
            three columns matching 'field_names'. If None, random data will be generated.

        field_names : list of str, default ['id1', 'id2', 'timestamp']
            Names of the columns in 'raw_data'. Must contain:
            - projected_layer (node set to analyze, e.g. 'id1')
            - other_layer (opposite node set, e.g. 'id2')
            - timestamp_col (time information)

        timeframe : str or int, default '%Y'
            Temporal resolution:
            - If 'time_type='actual'': a datetime format string (e.g. '%Y', '%Y-%m').
            - If 'time_type='intrinsic'': an integer defining the size of sequential bins.

        time_type : {'actual', 'intrinsic'}, default 'actual'
            How to interpret the timeframe:
            - 'actual' → uses calendar units (year, month, week, day).
            - 'intrinsic' → bins based on row order (useful for event streams).

        agg_func : callable, optional
            Custom aggregation function for interaction weights. Currently not implemented.

        distance_function : callable, default None
            Function used to compute distances between node vectors.

        dimensions : int, optional
            If provided, reduces positions into lower-dimensional space via, by deafult, PCA.

        reduction_function = callable, default None
            The user can specify what kind of reduction function should be applied. By defualt it's PCA.

        comparison : {'relative', 'absolute'}, default 'relative'
            Determines how distances are computed:
            - 'relative' → per timeframe independently.
            - 'absolute' → across all timeframes combined.

        num_rows : int, default 1000
            Number of rows to generate when simulating random data.

        n_layer_1 : int, default 3
            Number of unique nodes in the first layer (id1) for random data.

        n_layer_2 : int, default 20
            Number of unique nodes in the second layer (id2) for random data.

        start_dt : datetime, default datetime(2020, 1, 1)
            Start date for random timestamp generation.

        end_dt : datetime, default datetime(2025, 1, 1)
            End date for random timestamp generation.

        Attributes
        ----------
        projected_layer : str
            Column name representing the analyzed node set.
        other_layer : str
            Column name representing the secondary node set.
        timestamp_col : str
            Column name representing timestamps.
        biadjacency_matrix : pd.DataFrame or None
            Stores computed bipartite adjacency matrix.
        positions : pd.DataFrame or None
            Stores computed distance matrices.
        reduced_positions : pd.DataFrame or None
            Stores dimension-reduced positions.
        velocities : pd.DataFrame or None
            Stores computed velocities.

        Raises
        ------
        ValueError
            If 'field_names' is not a list of at least three elements.

        Example
        -------
        >>> from datetime import datetime
        >>> stdc = STDC(
        ...     field_names=['user', 'item', 'time'],
        ...     timeframe='%Y-%m',
        ...     time_type='actual',
        ...     dimensions=2
        ... )
        >>> print(stdc.raw_data.head())
            id1   id2   timestamp
        0   L1_0  L2_1  2020-02-15 12:45:33
        1   L1_1  L2_3  2020-06-10 08:22:14
        ...
        """

        # check for correct field input
        if not isinstance(field_names, list) or len(field_names) < 3 or type(field_names) is not list:
            raise ValueError("field_names must be a list with 3 ordered elements: [projected_layer, other_layer, timestamp_col]; default: ['id1', 'id2', 'timestamp']")
        
        # set up column names
        self.projected_layer = field_names[0]
        self.other_layer = field_names[1]
        self.timestamp_col = field_names[2]

        # global defaults
        self.__timeframe = timeframe
        self.__time_type = time_type
        self.__agg_func = agg_func
        self.__distance_function = distance_function
        self.__dimensions = dimensions
        self.__reduction_function = reduction_function
        self.__community_detection = community_detection
        self.__comparison = comparison

        # random data config
        self.__num_rows = num_rows
        self.__nlayer1 = n_layer_1
        self.__nlayer2 = n_layer_2
        self.__start_dt = start_dt
        self.__end_dt = end_dt

        # generate random data if no data is passed
        self.raw_data = raw_data.copy() if raw_data is not None else self.random_data_gen()  

    # --------------------------------
    # analysis & preprocessing methods
    # --------------------------------

    def random_data_gen(self):
        """
        Generate a random bipartite dataset for testing and demonstration.

        Creates a synthetic dataset with two layers ('id1', 'id2') and a timestamp. 
        This mimics a bipartite temporal graph structure (e.g., users interacting 
        with items over time).

        Returns
        -------
        pd.DataFrame
            DataFrame with three columns:
            - 'id1': entities from layer 1
            - 'id2': entities from layer 2
            - 'timestamp': datetime objects

        Example
        -------
        >>> stdc = STDC(num_rows=10, n_layer_1=2, n_layer_2=3)
        >>> df = stdc.random_data_gen()
        >>> print(df.head())
              id1   id2           timestamp
        0    L1_1  L2_2  2021-09-04 18:43:14
        1    L1_0  L2_0  2020-04-12 09:23:41
        ...
        """
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
    
    def calculate_timeframe(self, filter_always_present=True):
        """
        Add a 'timeframe' column to the raw dataset.

        Depending on the configuration, the time dimension can be binned in 
        either actual calendar units (year, month, week, day) or intrinsic units 
        (equal chunks based on row order).

        Returns
        -------
        pd.DataFrame
            The input 'raw_data' DataFrame with an added 'timeframe' column.

        Raises
        ------
        ValueError
            If 'time_type' and 'timeframe' argument types are incompatible.
        NotImplementedError
            If an unsupported time_type is provided.

        Example
        -------
        >>> stdc = STDC()
        >>> df = stdc.calculate_timeframe()
        >>> print(df[['timestamp', 'timeframe']].head())
                 timestamp timeframe
        0 2020-01-01 12:31:45     2020
        1 2020-02-15 09:22:17     2020
        ...
        """
        if self.__time_type == 'actual':
            if not isinstance(self.__timeframe, str):
                raise ValueError("If time_type is actual, timeframe must be a string.")
            self.raw_data['timeframe'] =  self.raw_data[self.timestamp_col].dt.strftime(self.__timeframe)
        elif self.__time_type == 'intrinsic':
            if not isinstance(self.__timeframe, (int)):
                raise ValueError("If time_type is intrinsic, timeframe must be an integer.")
            self.raw_data = self.raw_data.sort_values(by=[self.timestamp_col])
            self.raw_data['timeframe'] = range(len(self.raw_data))
            self.raw_data['timeframe'] = np.floor(self.raw_data['timeframe'] / self.__timeframe).astype(int)
        else:
            raise NotImplementedError("Not implemented yet.")

        if filter_always_present:
            n_timeframes = self.raw_data['timeframe'].nunique()
            tmp = self.raw_data.groupby(self.projected_layer)['timeframe'].nunique()
            layer1_always_present = tmp[tmp == n_timeframes].index.tolist()
            self.raw_data = self.raw_data[self.raw_data[self.projected_layer].isin(layer1_always_present)]
        return self.raw_data

    def calculate_biadjacency_matrix(self):
        """
        Construct a bipartite adjacency matrix across timeframes.

        Builds a matrix representation of interactions between 'projected_layer' 
        and 'other_layer', grouped by timeframe. Each entry contains the count 
        of interactions.

        Returns
        -------
        pd.DataFrame
            Biadjacency matrix with MultiIndex (projected_layer, timeframe) as rows 
            and 'other_layer' as columns.

        Example
        -------
        >>> stdc = STDC()
        >>> B = stdc.calculate_biadjacency_matrix()
        >>> print(B.head())
        other_layer         L2_0  L2_1  L2_2 ...
        id1  timeframe                          
        L1_0 2020               2     0     1
        L1_1 2020               0     1     3
        """
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
        """
        Compute distance matrices (relative or absolute) between nodes.

        Depending on the configuration:
        - relative: distances are computed per timeframe independently.
        - absolute: distances are computed across all timeframes combined.

        Returns
        -------
        pd.DataFrame
            Distance matrix (cosine by default), indexed by (node, timeframe).

        Example
        -------
        >>> stdc = STDC()
        >>> P = stdc.calculate_positions()
        >>> print(P.head())

        # Sample head output (distances; rows indexed by (node, timeframe), columns = node names):
        #                     L1_0      L1_1      L1_2
        # id1   timeframe
        # L1_0  2020         0.000000  0.278354  0.412097
        # L1_1  2020         0.278354  0.000000  0.365421
        # L1_2  2020         0.412097  0.365421  0.000000
        """
        if not hasattr(self, 'biadjacency_matrix'):
            self.calculate_biadjacency_matrix()

        if self.__comparison == 'relative':
            relative_cosine_dist_matrices = pd.DataFrame()

            for tf in self.biadjacency_matrix.index.levels[1].unique():
                tmp_raw_data = self.biadjacency_matrix.xs(tf, level='timeframe')
                multi_index = pd.MultiIndex.from_arrays([tmp_raw_data.index, [tf]*len(tmp_raw_data.index)], names=[self.projected_layer, 'timeframe'])
                if self.__distance_function is None:
                    tmp_cosine_dist_matrix = pd.DataFrame(
                        cosine_distances(tmp_raw_data),
                        index=multi_index,
                        columns=tmp_raw_data.index
                    )
                else:
                    tmp_cosine_dist_matrix = pd.DataFrame(
                        pairwise_distances(tmp_raw_data, metric = self.__distance_function),
                        index=multi_index,
                        columns=tmp_raw_data.index
                    )

                relative_cosine_dist_matrices = pd.concat([relative_cosine_dist_matrices, tmp_cosine_dist_matrix], axis=0)

            self.positions = relative_cosine_dist_matrices.fillna(0)
            return self.positions
        
        else:
            self.positions = pd.DataFrame(
                cosine_distances(self.biadjacency_matrix) if self.__distance_function is None
                    else pairwise_distances(self.biadjacency_matrix, metric = self.__distance_function),
                index=self.biadjacency_matrix.index,
                columns=self.biadjacency_matrix.index
                ).fillna(0)
            return self.positions
        
    
    def calculate_graphs(self):
        # check if positions exist
        if not hasattr(self, 'positions'):
            self.calculate_positions()

        self.graphs = {}

        # filter for time frame
        for tf in self.positions.index.levels[1].unique():
            filtered = 1 - self.positions.xs(tf, level='timeframe')
            #filter for columns as well, if comparison is absolute
            if self.__comparison == 'absolute':
                filtered = filtered.xs(tf, level='timeframe', axis = 1)

            # construct graph
            g = gt.Graph(g=sp.coo_array(filtered.values), directed=False)
            gt.remove_self_loops(g)
            self.graphs[tf] = g

        return self.graphs

    def calculate_communities(self):
        # check for existence of graphs
        if not hasattr(self, 'graphs'):
            self.calculate_graphs()

        #calculates for communities in every single timeframe
        self.communities = {}

        # for every timeframe: graph, calculate community detection
        for tf, graph in self.graphs.items():
            # calculate communities
            if self.__community_detection == "SBM":
                state = gt.minimize_blockmodel_dl(graph, state_args=dict(recs=[graph.ep.weight], rec_types=["real-exponential"], deg_corr=True))
                self.communities[tf] = state.b #state.b should now be the equivalent of state.get_blocks() and returns a vertex property map
            elif self.__community_detection == "Leiden":
                igg = ig.Graph.from_graph_tool(graph)
                ig_partition = igg.community_leiden(objective_function='modularity',weights=igg.es['weight'], n_iterations=10)
                gt_partition = graph.new_vertex_property("int")
                for e, part in enumerate(ig_partition):
                    for node in part:
                        gt_partition[node] = e
                self.communities[tf] = gt_partition
            else:
                print("Other community detection methods have not been implemented yet.")
                # state = gt.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.weight], rec_types=["real-exponential"], deg_corr=True))
                pass
        return self.communities

    def calculate_modularities(self):
        #calculates modularity on each of the graph
        if not hasattr(self, 'communities'):
            self.calculate_communities()

        self.modularities = pd.DataFrame()
        
        # for every graph from the community detection function that stores the graphs with the communities
        for tf, community in self.communities.items():
            # Check if 'weight' edge property exists
            if self.graphs[tf].num_edges() > 0:
                modularity = gt.modularity(self.graphs[tf], community, weight=self.graphs[tf].ep.weight)
                self.modularities = pd.concat([self.modularities, pd.DataFrame({'timeframe': [tf], 'modularity': [modularity]})], ignore_index=True)
            else:
                print(f"Graph at timeframe {tf} has no edges. Modularity is not calculated.")

        return self.modularities

    def calculate_aligned_modularities(self):
        # does the same as calculate_modularities, but aligns modularity the same way as the velocities
        # return average of modularities of two consecutive timeframes
        if not hasattr(self, 'modularity'):
            self.calculate_modularities()

        self.aligned_modularities = pd.DataFrame()

        # sort timeframes
        timeframes = sorted(self.modularities['timeframe'])

        # iterate over consecutive pairs
        for t1, t2 in zip(timeframes[:-1], timeframes[1:]):
            mod1 = self.modularities.loc[self.modularities['timeframe'] == t1, 'modularity'].values[0]
            mod2 = self.modularities.loc[self.modularities['timeframe'] == t2, 'modularity'].values[0]
            avg_modularity = (mod1 + mod2) / 2
            self.aligned_modularities = pd.concat([self.aligned_modularities, pd.DataFrame({'t1': [t1], 't2': [t2], 'modularity': [avg_modularity]})],ignore_index=True)

        return self.aligned_modularities

    def calculate_reduced_positions(self, verbose = False):
        """
        Apply dimensionality reduction (optional) on the distance matrix.

        Uses PCA to reduce 'positions' into a lower-dimensional embedding 
        (e.g., 2D for visualization). If 'dimensions' is None, returns original 
        positions.

        Returns
        -------
        pd.DataFrame
            Reduced positions DataFrame.

        Notes
        -----
        - Saves explained variance ratio to 'self.explained_variance_ratio_' if PCA is applied.

        Example
        -------
        >>> stdc = STDC(dimensions=2)
        >>> R = stdc.calculate_reduced_positions()
        >>> print(R.head())
        # (If verbose = True, the method prints: Explained variance ratio: [0.455609 0.420137])
        # Sample head output (rows indexed by (node, timeframe); columns are PCA components 0 and 1):
        #                       0         1
        # id1   timeframe
        # L1_0  2020     0.312345 -0.123456
        # L1_1  2020    -0.045678  0.543210
        # L1_2  2020     0.234567 -0.345678
        # L1_0  2021     0.123456  0.234567
        # L1_1  2021    -0.234567  0.111111
        """
        if not hasattr(self, 'positions'):
            self.calculate_positions()

        if self.__dimensions is not None:
            if self.__reduction_function is None:
                pca = PCA(n_components=self.__dimensions)
                self.reduced_positions = pd.DataFrame(pca.fit_transform(self.positions), index=self.positions.index).fillna(0)
                self.explained_variance_ratio_ = pca.explained_variance_ratio_
                if verbose:
                    print("Explained variance ratio:", self.explained_variance_ratio_)
            else:
                u = umap.UMAP(n_components=self.__dimensions, metric = self.__distance_function if self.__distance_function is not None else 'cosine')
                self.reduced_positions = pd.DataFrame(u.fit_transform(self.positions), index=self.positions.index).fillna(0)
            return self.reduced_positions
        else:
            self.reduced_positions = self.positions
            return self.reduced_positions
        
    def calculate_aligned_reduced_positions(self):
        """
        Calculate and align reduced positions to match the timeframes of velocities.

        This method computes the average of consecutive reduced positions for each node,
        effectively aligning the reduced positions with the timeframes used for velocity calculations.
        The first timeframe data point is omitted to ensure the resulting DataFrame is directly comparable
        to the velocities DataFrame, as both will now reference the same time intervals.

        If reduced positions have not yet been calculated, this method will call `calculate_reduced_positions()`
        to generate them.

        The resulting DataFrame uses a MultiIndex with the following levels:
            - The projected layer (e.g., node or entity identifier)
            - 't1': the starting time index of the interval
            - 't2': the ending time index of the interval

        The aligned reduced positions are stored in `self.aligned_reduced_positions` and also returned.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the aligned reduced positions, indexed by the projected layer,
            't1', and 't2', and aligned with the velocity timeframes.

        Example
        -------
        >>> stdc = STDC(dimensions=2)
        >>> stdc.calculate_reduced_positions()
        >>> aligned = stdc.calculate_aligned_reduced_positions()
        >>> print(aligned.head())
        # Sample head output (index = (node, t1, t2); columns = per-component averages):
        #                          0         1
        # id1   t1    t2
        # L1_0  2020  2021   -0.226492  0.151122
        #       2021  2022   -0.223539 -0.068125
        #       2022  2023   -0.113316 -0.096172
        #       2023  2024   -0.038281 -0.053525
        """
        if not hasattr(self, 'reduced_positions'):
            self.calculate_reduced_positions()

        aligned_positions = pd.DataFrame()

        for node in self.reduced_positions.index.get_level_values(self.projected_layer).unique():
            tmp = self.reduced_positions.xs(node, level=self.projected_layer).sort_index()
            avg = (tmp.shift(-1) + tmp) / 2
            avg = avg.iloc[:-1]
            avg.index = pd.MultiIndex.from_arrays(
            [[node]*len(avg), tmp.index[:-1], tmp.index[1:]],
            names=[self.projected_layer, 't1', 't2']
            )
            aligned_positions = pd.concat([aligned_positions, avg], axis=0)

        self.aligned_reduced_positions = aligned_positions
        return self.aligned_reduced_positions
    
    def calculate_velocities(self):
        """
        Compute node velocities between consecutive timeframes.

        Velocity is calculated as the difference in reduced position vectors 
        across consecutive timeframes.

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame with levels (node, t1, t2), where each row 
            contains the velocity vector for that transition.

        Example
        -------
        >>> stdc = STDC(dimensions=2)
        >>> stdc.calculate_reduced_positions()
        >>> V = stdc.calculate_velocities()
        >>> print(V.head())

        # Sample head output (index = (node, t1, t2); columns = per-component differences), if we use time_type = 'actual' and dimensions=2:
        #                          0         1
        # id1   t1    t2
        # L1_0  2020  2021    0.18901  0.35712
        # L1_1  2020  2021   -0.05890  0.12034
        # L1_2  2020  2021    0.00000 -0.04560
        # L1_0  2021  2022   -0.12345  0.06780
        # L1_1  2021  2022    0.04560 -0.00980
        """
        if not hasattr(self, 'reduced_positions'):
            self.calculate_reduced_positions()

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
        Compute velocity magnitudes between consecutive timeframes using 
        a custom distance function.

        Unlike 'calculate_velocities' (which uses raw vector differences), 
        this method calculates the distance between reduced positions at 
        consecutive timeframes for each node.

        Returns
        -------
        pd.DataFrame
            Columns: ['Node', 't1', 't2', 'Velocity']

        Example
        -------
        >>> stdc = STDC(dimensions=2)
        >>> stdc.calculate_reduced_positions()
        >>> CDV = stdc.calculate_custom_distance_velocities()
        >>> print(CDV.head())
             Node    t1    t2  Velocity
        0   L1_0  2020  2021  0.2345
        """
        if not hasattr(self, 'reduced_positions'):
            self.calculate_reduced_positions()

        if self.positions.index.nlevels == 2:
            timeframes = self.positions.index.get_level_values("timeframe").unique().sort_values()
        elif self.positions.index.nlevels == 1 and 'timeframe' in self.positions.columns:
            timeframes = self.positions['timeframe'].unique().sort()
        else:
            raise ValueError("Positions index must contain 'timeframe'.")
        
        velocity_records = []

        for i in range(len(timeframes) - 1):
            current_tf = timeframes[i]
            next_tf = timeframes[i + 1]

            try:
                current_data = self.positions.xs(current_tf, level="timeframe")
                next_data = self.positions.xs(next_tf, level="timeframe")
            except Exception:
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
                    if self.__distance_function is None:
                        velocity = cosine_distances([vec1], [vec2])[0][0]
                    else:
                        raise NotImplementedError("Other velocity distance functions have not been implemented yet.")
                    
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

    def calculate_basic_ts_stats(self, temp=True, vol=True, vel_CoM=True): 
        if not hasattr(self, 'aligned_reduced_positions'):
            self.calculate_aligned_reduced_positions()
        if not hasattr(self, 'velocities'):
            self.calculate_velocities()

        self.p_stats = self.aligned_reduced_positions.groupby(['t1','t2']).agg(['mean','var','count'])
        self.v_stats = self.velocities.groupby(['t1','t2']).agg(['mean','var','count'])
        return self.p_stats, self.v_stats

    def calculate_thermodyn_ts_stats(self):
        if not hasattr(self, 'p_stats') or not hasattr(self, 'v_stats'):
            self.calculate_basic_ts_stats()
        if not hasattr(self, 'aligned_modularities'):
            self.calculate_aligned_modularities()
        
        vol_ts = np.sqrt(self.p_stats.xs('var', axis = 1, level = 1 + (self.__dimensions == None))).prod(axis = 1)
        temp_ts = self.v_stats.xs('var', axis = 1, level = 1 + (self.__dimensions == None)).sum(axis = 1)
        vcom_ts = np.sqrt(np.power(self.v_stats.xs('mean', axis = 1, level = 1 + (self.__dimensions == None)), 2).sum(axis = 1)) # V = (V_x, V_y, ...) -> |V| = sqrt(V_x^2 + V_y^2 + ...)
        counts_ts = self.p_stats['count'].values
        self.thermo_stats = pd.DataFrame({'Vol': vol_ts, 'Temp': temp_ts, 'V_CoM': vcom_ts, 'Mod': self.aligned_modularities.set_index(['t1','t2'])['modularity'], 'CNT': counts_ts})
        return self.thermo_stats

    # --------------------------------
    # visualisation methods
    # --------------------------------
    def plot_center_of_mass_trajectory(self):
        if not hasattr(self, 'p_stats'):
            self.calculate_basic_ts_stats()
        if self.p_stats.xs('mean', axis = 1, level = 1 + (self.__dimensions == None)).shape[1] > 2:
            print("Warning: More than 2 dimensions detected. Plotting only the first two dimensions.")
        x = (self.p_stats.xs('mean', level = 1 + (self.__dimensions == None), axis = 1)).iloc[:, :1].values.flatten()
        y = (self.p_stats.xs('mean', level = 1 + (self.__dimensions == None), axis = 1)).iloc[:, 1:2].values.flatten()
        x_err = np.sqrt((self.p_stats.xs('var', level = 1 + (self.__dimensions == None), axis = 1)).iloc[:, :1]/
                        (self.p_stats.xs('count', level = 1 + (self.__dimensions == None), axis = 1)).iloc[:, :1]).values.flatten()
        y_err = np.sqrt((self.p_stats.xs('var', level = 1 + (self.__dimensions == None), axis = 1)).iloc[:, 1:2]/
                        (self.p_stats.xs('count', level = 1 + (self.__dimensions == None), axis = 1)).iloc[:, 1:2]).values.flatten()

        ps = plt.scatter(x, y, c=np.linspace(0, 1, self.p_stats.shape[0]), vmin=0, vmax=1, cmap=plt.cm.rainbow)

        colors = plt.cm.rainbow(np.linspace(0, 1, self.p_stats.shape[0]))
        plt.errorbar(x, y, xerr=x_err, yerr=y_err,
                     ecolor=colors, alpha=0.5)
        plt.colorbar(ps, label='Time progression')

        plt.quiver( x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],    # vector components
            scale_units='xy', angles='xy', scale=1, color='black', width=0.01, alpha=0.5)
        
    def plot_reduced_positions_animation(self, figsize=(6, 6), interval=1000, fps=1, save_path=None):
        """
        Create an animated visualization of reduced positions over time.

        Generates an animation showing how node positions evolve across timeframes
        in the reduced dimensional space (PCA, UMAP, etc.).

        Parameters
        ----------
        figsize : tuple of int, default (6, 6)
            Figure size in inches (width, height).
        interval : int, default 1000
            Delay between frames in milliseconds.
        fps : int, default 1
            Frames per second when saving as GIF.
        save_path : str, optional
            If provided, saves the animation as a GIF to this path.
            Requires 'pillow' package.

        Returns
        -------
        IPython.display.HTML
            Animated HTML display for Jupyter notebooks.

        Example
        -------
        >>> stdc = STDC(dimensions=2, reduction_function='umap')
        >>> stdc.plot_reduced_positions_animation(save_path="evolution.gif")
        """
        if not hasattr(self, 'reduced_positions'):
            self.calculate_reduced_positions()

        # Determine axis labels based on reduction function
        if self.__reduction_function is None:
            reduction_name = "PCA"
        else:
            reduction_name = "UMAP"

        # Sort by timeframe
        sorted_reduced_positions = self.reduced_positions.sort_index(level='timeframe')

        # Get unique timeframes
        timeframes = sorted_reduced_positions.index.get_level_values('timeframe').unique().sort_values()

        # Prepare figure
        fig, ax = plt.subplots(figsize=figsize)
        scat = ax.scatter([], [], s=50, alpha=0.6)

        ax.set_xlabel(f"{reduction_name} 1")
        ax.set_ylabel(f"{reduction_name} 2")
        ax.set_title(f"{reduction_name} Evolution Over Time")

        x_min, x_max = sorted_reduced_positions.iloc[:, 0].min(), sorted_reduced_positions.iloc[:, 0].max()
        y_min, y_max = sorted_reduced_positions.iloc[:, 1].min(), sorted_reduced_positions.iloc[:, 1].max()

        # Set limits
        x_lowlim = x_min - 0.05 * (x_max - x_min)
        x_uplim = x_max + 0.05 * (x_max - x_min)
        y_lowlim = y_min - 0.05 * (y_max - y_min)
        y_uplim = y_max + 0.05 * (y_max - y_min)
        
        ax.set_xlim(x_lowlim, x_uplim)
        ax.set_ylim(y_lowlim, y_uplim)
        ax.grid(True, alpha=0.3)

        # Update function
        def update(frame):
            t = timeframes[frame]
            subset = sorted_reduced_positions.xs(t, level='timeframe')
            scat.set_offsets(subset.iloc[:, [0, 1]].values)
            ax.set_title(f"{reduction_name} Evolution — Timeframe {t}")
            return scat,

        # Create animation
        ani = FuncAnimation(fig, update, frames=len(timeframes), 
                        interval=interval, blit=False, repeat=True)

        # Save to GIF if path provided
        if save_path is not None:
            ani.save(save_path, writer="pillow", fps=fps)
            print(f"Animation saved to {save_path}")

        # Display inline
        plt.close(fig)  # prevent static image display
        return HTML(ani.to_jshtml())

    def plot_pca_interactive():
        pass

    def plot_combined_graph():
        pass

    def plot_kde_overlay():
        pass