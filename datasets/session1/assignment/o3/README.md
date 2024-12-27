# UCI Mechanical Analysis Data Set


1 - instance - instance indicator

1 - component - component number (integer)

2 - sup - support in the machine where measure was taken (1..4)

3 - cpm - frequency of the measure (integer)

4 - mis - measure (real)

5 - misr - earlier measure (real)

6 - dir - filter, type of the measure and direction:
{vo=no filter, velocity, horizontal,
va=no filter, velocity, axial,
vv=no filter, velocity, vertical,
ao=no filter, amplitude, horizontal,
aa=no filter, amplitude, axial,
av=no filter, amplitude, vertical,
io=filter, velocity, horizontal,
ia=filter, velocity, axial,
iv=filter, velocity, vertical}

7 - omega - rpm of the machine (integer, the same for components of one example)

8 - class - classification (1..6, the same for components of one example)

9 - comb. class - combined faults

10 - other class - other faults occuring

Acknowledgements
Data Source: https://archive.ics.uci.edu/ml/datasets/Mechanical+Analysis