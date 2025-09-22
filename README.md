# STDC – Social Thermodynamics Class

STDC is a Python package for analyzing **temporal bipartite networks**.  
It extends beyond static graph libraries by computing **graph dynamics**, including:  
- Time-binned bipartite adjacency matrices  
- Distance matrices (relative or absolute)  
- Dimensionality reduction of positions  
- Node velocities across timeframes  

---

## Installation

```bash
git clone https://github.com/yourusername/stdc.git
cd stdc
```

## Tutorial
```
from stdc import STDC

# Initialize with synthetic data
stdc = STDC(dimensions=2, timeframe='%Y')

# Optionally, use your own data
# raw_data = pd.DataFrame({'id1': ..., 'id2': ..., 'timestamp': ...})
# stdc = STDC(raw_data=raw_data, dimensions=2)

# 1. Generate or load data
df = stdc.raw_data
print(df.head())
# Sample output:
#       id1   id2           timestamp
# 0    L1_0  L2_1  2020-02-15 12:45:33
# 1    L1_1  L2_3  2020-06-10 08:22:14
# ...

# 2. Add timeframe bins
df = stdc.calculate_timeframe()
print(df[['timestamp', 'timeframe']].head())
# Sample output:
#            timestamp timeframe
# 0 2020-01-01 12:31:45     2020
# 1 2020-02-15 09:22:17     2020
# ...

# 3. Build biadjacency matrix
B = stdc.calculate_biadjacency_matrix()
print(B.head())
# Sample output:
# other_layer         L2_0  L2_1  L2_2 ...
# id1  timeframe                          
# L1_0 2020               2     0     1
# L1_1 2020               0     1     3

# 4. Compute positions (distance matrices)
P = stdc.calculate_positions()
print(P.head())
# Sample output:
#                     L1_0      L1_1      L1_2
# id1   timeframe
# L1_0  2020         0.000000  0.278354  0.412097
# L1_1  2020         0.278354  0.000000  0.365421
# L1_2  2020         0.412097  0.365421  0.000000
# L1_0  2021         0.000000  0.123456  0.289012
# L1_1  2021         0.123456  0.000000  0.401234

# 5. Reduce dimensions (PCA → 2D)
R = stdc.calculate_reduced_positions()
print(R.head())
# Sample output (prints explained variance ratio too):
#                       0         1
# id1   timeframe
# L1_0  2020     0.312345 -0.123456
# L1_1  2020    -0.045678  0.543210
# L1_2  2020     0.234567 -0.345678
# L1_0  2021     0.123456  0.234567
# L1_1  2021    -0.234567  0.111111

# 6. Calculate velocities (vector differences between timeframes)
V = stdc.calculate_velocities()
print(V.head())
# Sample output:
#                          0         1
# id1   t1    t2
# L1_0  2020  2021    0.18901  0.35712
# L1_1  2020  2021   -0.05890  0.12034
# L1_2  2020  2021    0.00000 -0.04560
# L1_0  2021  2022   -0.12345  0.06780
# L1_1  2021  2022    0.04560 -0.00980

# 7. Calculate custom distance velocities (distance function between timeframes)
CDV = stdc.calculate_custom_distance_velocities()
print(CDV.head())
# Sample output:
#     Node    t1    t2  Velocity
# 0   L1_0  2020  2021  0.2345
# 1   L1_1  2020  2021  0.1456
# 2   L1_2  2020  2021  0.0987
# 3   L1_0  2021  2022  0.1678
# 4   L1_1  2021  2022  0.0564
```
