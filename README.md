# Jellyfish-vs-FatTree
Homework #1 for the course "Networking for Big Data and Data Centers" at La Sapienza University of Rome

## Content
>- `main.py`: python script that calls the functions and the classes from the modules below and saves the resulting plots in images (png);
>- `functions.py`: python script file which contains all the functions for building graphs, make simulations and evaluating performances;
>- `topologies.py`: python module where we implemented the classes for the used topologies.

## Used technologies:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)

## Brief description
In the first part of the homework we define functions for creating the graph randomly by two techniques:
>- r-regular graph;
>- Erdòs-Rényi.

We then implement several techniques to evaluate whether the created random graphs are connected:
1. we varify the reachability of all nodes with a **breadth-first search** algorithm;
2. we verify that the **Laplacian matrix** associated with the graph has a null eigenvalue of multiplicity 1;
3. we verify that the **irreducibility** of the adjacency matrix (A) associated with the graph by solving the following inequality :
$$I + A+ A^2+ ... + A^n > 0$$

Finally, we implement from scratch the Jellyfish and Fat-tree topologies and compare their scalability performance.

Fat-tree            |  Jellyfish
:-------------------------:|:-------------------------:
<img width="490" alt="fat" src="https://user-images.githubusercontent.com/93355495/234046202-18d011f7-b848-4ec2-b6b6-b4193ad2d8e2.png"> | <img width="490" alt="jelly" src="https://user-images.githubusercontent.com/93355495/234046309-ee59d22a-ee97-467d-a5ba-06b2656240ed.png">
<br />

## Simulation and parformance evaluation
We evaluate the complexity, and thus the level of efficiency, of the methods listed above for studying the connectivity of a graph by considering two possible metrics.

Running time             |  Bytes occupied in RAM for execution
:-------------------------:|:-------------------------:
![complex_comparis](https://user-images.githubusercontent.com/93355495/233846103-545c0c71-32fc-4e9f-8690-c9fd9c186d53.png) | ![complex_comparis2](https://user-images.githubusercontent.com/93355495/233846168-4f657c08-81a3-4d01-bbeb-f5c1206731cb.png)
<br />

Then, let $p_c(G)$ denote the probability that a graph G is connected. By running Monte Carlo simulations, we estimate $p_c(G)$ and produce two curve plots:

1. $p_c(G)$ vs. $p$ for Erdòs-Rényi graphs with K = 100 number of nodes.
<p align=center>
<img alt="ERgraph" src="https://user-images.githubusercontent.com/93355495/233846382-fd9ac907-f75f-46da-b338-7107a931b0fc.png">
</p>

2. $p_c(G)$ vs. $K$, for K ≤ 100, for r-regular random graphs with r = 2 and r = 8.
<p align=center>
<img alt="Rregular" src="https://user-images.githubusercontent.com/93355495/233846396-1f99dc80-00c5-46ae-8e40-4ada8d2c1e6e.png">
</p>

<br />

Consider that if the job is split into N parallel tasks, that are run over N servers, then:
>- each task takes a time $T_0 + X_i$, where $X_i \sim Exp( \lambda= \frac N {E[X]})$ , X r.V. for the baseline job running time
>- each server receives an amount of input data $L_f /N$ ($L_f$: lenght of the baseline input file).
>- amount of output data produced by each server is $L_{o,i} \sim Unif(0, 2L_o/N)$ ($L_o$: lenght of the baseline output file)
>- Data is transferred to and from server i via a TCP connection between server A and server i, having average throughput given by: 
$$\theta_i = C \cdot \frac {1/T_i} {\sum_j T_j}$$
where $T_i=2 \tau h_i$ is the RTT between the origin server A and server i, $h_i$ is the number of hops between server A and server i and C is the capacity of each link of the DC network.

In the end we plot:

>- the mean response time $E[R]$ as a function of $N$ for $1 \leq N \leq 10000$. Let $R_baseline$ be the response time in case only server A is used, we normalize w.r.t. $ E[R_baseline]$
>- the Job running cost $S$ as a function of $N$ for  $1 \leq N \leq 10000$ (normalize $S$ with respect to $S_baseline$).

<img width="1184" alt="TimeAndCost" src="https://user-images.githubusercontent.com/93355495/234045317-4b00ed8d-5831-43b8-80c3-39582a2ad325.png">

