# STDL – Social Thermodynamics Library

## Introduction
STDL is a Python package for analyzing **temporal bipartite networks**.  
It extends beyond static graph libraries by computing **graph dynamics**, including:  
- Time-calculated bipartite adjacency matrices  
- Distance matrices (relative or absolute)  
- Dimensionality reduction of positions  
- Node velocities across timeframes  

### Credits
Pietro Gravino, Vito D. P. Servedio, Giulio Prevedello, Juraj Simkovic

## Tutorial

**Initialize STDC object**  
Create an STDC (Social Thermodynamics Class) object with 2D reduced positions and yearly timeframes. The comparison is specified as 'absolute'.
```
In [1]: from stdc import STDC
In [2]: stdc = STDC(dimensions=2, timeframe='%Y', comparison='absolute')
In [3]: stdc.raw_data.head()
Out[3]:
      id1   id2           timestamp
0    L1_0  L2_1  2020-02-15 12:45:33
1    L1_1  L2_3  2020-06-10 08:22:14
2    L1_2  L2_0  2020-09-03 16:11:22
3    L1_0  L2_2  2021-01-15 10:44:55
4    L1_1  L2_4  2021-03-21 09:20:18
```

**Compute distance matrix**  
Compute the absolute distance matrices combining all timeframes based on the user-specified timeframe.
```
In [4]: stdc.calculate_positions()
In [5]: stdc.positions.head()
Out[5]:
id1   timeframe   L1_0_2020  L1_0_2021  L1_0_2022  L1_1_2020  L1_1_2021    ...
L1_0 2020           0.000000   0.268102   0.387436   0.255528   0.222681   
     2021           0.268102   0.000000   0.275309   0.230118   0.243040   
     2022           0.387436   0.275309   0.000000   0.337972   0.231917  
L1_1 2020           0.255528   0.230118   0.337972   0.000000   0.223293   
     2021           0.222681   0.243040   0.231917   0.223293   0.000000   
     ...
```

**Calculate reduced positions**  
Apply dimensionality reduction (PCA by default) when calculating positions.
```
In [6]: stdc.calculate_reduced_positions()
In [7]: stdc.reduced_positions.head()
Out[7]:
                      0         1
id1   timeframe
L1_0  2020     0.312345 -0.123456
L1_0  2021     0.123456  0.234567
L1_0  2022    -0.098765  0.187654
L1_1  2020    -0.045678  0.543210
L1_1  2021    -0.234567  0.111111
```

**Calculate velocities**  
Compute node velocities as differences of reduced positions between consecutive timeframes.
```
In [8]: stdc.calculate_velocities()
In [9]: stdc.velocities.head()
Out[9]:
                          0         1
id1   t1    t2
L1_0  2020  2021    0.18901  0.35712
L1_0  2021  2022   -0.12345  0.06780
L1_1  2020  2021   -0.05890  0.12034
L1_1  2021  2022    0.04560 -0.00980
L1_2  2020  2021    0.00000 -0.04560
```

**Show biadjacency matrix**  
By now, the biadjacency matrix has been calculated as a necessary step, when providing the distances matrix. The user can retrieve it simply by accessing it as an object attribute.
```
In [10]: stdc.biadjacency_matrix.head()
Out[10]:
other_layer         L2_0  L2_1  L2_2  L2_3  L2_4
id1  timeframe
L1_0 2020               1     1     0     0     0
L1_1 2020               0     0     0     1     0
L1_2 2020               1     0     0     0     0
L1_0 2021               0     0     1     0     0
L1_1 2021               0     0     0     0     1
```

**Show raw_data columns**  
The timeframe column has also been created and calculated as a necessary step of the previous analyses.
```
In [11]: stdc.raw_data[['timestamp', 'timeframe']].head()
Out[11]:
            timestamp timeframe
0 2020-02-15 12:45:33     2020
1 2020-06-10 08:22:14     2020
2 2020-09-03 16:11:22     2020
3 2021-01-15 10:44:55     2021
4 2021-03-21 09:20:18     2021
```
