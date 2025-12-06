# STDL – Social Thermodynamics Library

STDL is a Python package for analyzing **temporal bipartite networks**. It extends beyond static graph libraries by computing **graph dynamics** such as entity positions, velocities between consecutive timeframes, graphs, communities, modularities, visualisations and thermodynamic statistics.

### Credits
**DESIGN and DEVELOPMENT**: Pietro Gravino (Sony CSL Paris), Juraj Simkovic (Vienna University of Technology)
<br/>
**SUPPORT and SUPERVISION**: Pietro Gravino (Sony CSL Paris), Giulio Prevedello (Sony CSL Paris), Vito Servedio (Complexity Science Hub Vienna)

### Installation

```bash
!conda install --yes --file requirements.txt
```

### Usage

```python
from STDL import STDC

stdc = STDC(
    raw_data=your_data,
    field_names=['id1', 'id2', 'timestamp'],
    timeframe='%Y-%m',
    time_type='actual',
    community_detection='Leiden',
    reduction_function='UMAP'
)

stdc.calculate_reduced_positions()
stdc.calculate_velocities()

stdc.plot_reduced_positions_animation()
stdc.plot_center_of_mass_trajectory()
```
For a more extensive tutorial, please refer to the [example_notebook.ipynb](example_notebook.ipynb).

## Documentation

### Initialization

Create an `STDC` instance by passing the optional arguments:

* ***raw_data : pd.DataFrame, default None***  
  Input dataset with timestamps. Must include at least three columns matching `field_names`.  
  If `None`, random data will be generated.

* ***field_names : list of str, default ['id1', 'id2', 'timestamp']***  
  Names of the columns in `raw_data`. Must contain:  
  - `main_layer` — node set to analyze (e.g., `id1`)  
  - `other_layer` — opposite node set (e.g., `id2`)  
  - `datetime_col` — timestamp column (must be of type datetime)

* ***timeframe : str or int, default '%Y'***  
  Timeframe to analyze:  
  - For `time_type='actual'`: a datetime format string (e.g., `'%Y'`, `'%Y-%m'`)  
  - For `time_type='intrinsic'`: an integer defining the size of bins

* ***time_type : {'actual', 'intrinsic'}, default 'actual'***  
  Type of time analysis:  
  - `'actual'` - uses calendar units (year, month, week, day)  
  - `'intrinsic'` - count-based bins

* ***distance_function : callable, default None***  
  Function used to compute distances between node vectors.  
  If `None`, use cosine distance.

* ***dimensions : int, default None***  
  If provided, reduces the node positions into the specified number of dimensions.

* ***reduction_function : callable, default None***  
  Choose dimensionality-reduction function.  
  If `None`, PCA is used.

* ***community_detection : callable, default 'Leiden'***  
  Choose community detection algorithm (e.g. Leiden or SBM).  
  If `None`, Leiden is used.

* ***comparison : {'relative', 'absolute'}, default 'relative'***  
  Determines how distances are computed:  
  - `'relative'` - compute positions within the same timeframe
  - `'absolute'` - computed positions across all timeframes

* ***num_rows : int, default 1000***  
  Number of rows to generate when simulating random data.

* ***n_layer_1 : int, default 3***  
  Number of unique nodes in the first layer (`id1`) for random data.

* ***n_layer_2 : int, default 20***  
  Number of unique nodes in the second layer (`id2`) for random data.

* ***start_dt : datetime, default datetime(2020, 1, 1)***  
  Start date for random timestamp generation.

* ***end_dt : datetime, default datetime(2025, 1, 1)***  
  End date for random timestamp generation.


### Methods

**Basic Functionality**
```
def random_data_gen(self):

    """
    Generate synthetic timestamped bipartite data.
    Outputs main layer, opposite layer and a datetime column.
    """

def calculate_timeframe(self, filter_always_present=True):

    """
    Assign each record in the dataset to a timeframe bin specified by user.
    The bins are by default allocated by year.

    Parameters
    ----------
    filter_always_present : bool, optional
        Remove main layer entities which are not present in all timeframes.
    """

def calculate_biadjacency_matrix(self):

    """
    Output a matrix where a row is by default the count of the interactions
    between an entity of the main layer in a certain timeframe, with all other entities in the other layer.
    """

def calculate_positions(self):

    """
    Compute node positions based on the biadjacency matrix. By default, positions are computed as cosine distances.
    """

def calculate_reduced_positions(self, verbose=False):

    """
    Compute node positions and apply dimension reduction. By default, PCA.

    Parameters
    ----------
    verbose : bool, optional
        If True, PCA prints the explained variance ratio.
    """

def calculate_graphs(self):

    """
    Generate graphs from node positions for each timeframe.
    """

def calculate_communities(self):

    """
    Detect communities within each graph.
    """

def calculate_modularities(self):

    """
    Compute modularity scores for each graph.
    """

def calculate_aligned_modularities(self):

    """
    Compute modularities as average of two consecutive timeframes.
    """

def calculate_velocities(self):

    """
    Compute velocities as differences between node positions in consecutive timeframes.
    """
```
**Statistics**
```
def calculate_basic_ts_stats(self):

    """
    Calculate basic time-series statistics for node positions.
    Return mean positions, variance, and count for consecutive timeframes.
    """

def calculate_thermodyn_ts_stats(self):

    """
    Compute thermodynamic statistics such as volume, temperature, etc.
    """
```
**Visualisations**
```

def plot_center_of_mass_trajectory(self, groups=None):

    """
    Visualize movement of the center of mass, which is represented by the mean and the corresponding error bars.
    
    Parameters
    ----------
    groups : dict, optional
        Track movement of specific groups.
    """
```
<img width="805" height="511" alt="stdc_i_trajectory" src="https://github.com/user-attachments/assets/74319cb8-e37a-48f2-951c-16d5e66303b5" />

```
def plot_reduced_positions_animation(self, figsize=(6, 6), interval=1000, fps=1, save_path=None, labels=None):

    """
    Uses the reduced dimensionality positions and creates an evolution over all timeframes.

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
    labels : dict, optional
        Label and track movement of specific nodes.
    """
```
![pca_evolution](https://github.com/user-attachments/assets/d02c2fdd-669d-4516-875b-ab8dc4fd4180)
