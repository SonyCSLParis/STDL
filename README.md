# STDC – Social Thermodynamics Class

STDC is a Python package for analyzing **temporal bipartite networks**.  
It extends beyond static graph libraries by computing **graph dynamics**, including:  
- Time-calculated bipartite adjacency matrices  
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
**Initialize the STDC object**
Create an STDC object with 2D reduced positions and yearly timeframes. The comparison is specified as 'absolute'.
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
**Add timeframe bins**
Compute the absolute distance matrices combining all timeframes based on the user-specified timeframe.

```
In [4]: stdc.calculate_positions()
In [5]: stdc.positions.head()
Out[5]:
id1   timeframe   L1_0_2020  L1_0_2021  L1_0_2022  L1_1_2020  L1_1_2021  L1_1_2022  L1_2_2020  L1_2_2021  L1_2_2022
L1_0 2020           0.000000   0.268102   0.387436   0.255528   0.222681   0.337972   0.213329   0.179716   0.278857
L1_0 2021           0.268102   0.000000   0.275309   0.230118   0.243040   0.231917   0.260118   0.175786   0.141949
L1_0 2022           0.387436   0.275309   0.000000   0.337972   0.231917   0.000000   0.338422   0.300255   0.350229
L1_1 2020           0.255528   0.230118   0.337972   0.000000   0.223293   0.152172   0.139158   0.133469   0.265611
L1_1 2021           0.222681   0.243040   0.231917   0.223293   0.000000   0.188031   0.175786   0.179871   0.189565

```
