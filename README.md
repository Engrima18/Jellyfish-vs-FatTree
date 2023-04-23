# Jellyfish-vs-FatTree
Homework #1 for the course "Networking for Big Data and Data Centers" at La Sapienza University of Rome

## Content
>- `main.py`: python script that calls the functions and the classes from the modules below and saves the resulting plots in images (png);
>- `functions.py`: python script file which contains all the functions for building graphs, make simulations and evaluating performances;
>- `topologies.py`: python module where we implemented the classes for the used topologies.

## Brief description
In the first part of the homework we define functions for creating the graph randomly by two techniques:
>- r-regular graph;
>- Erdòs-Rényi.

We then implement several techniques to evaluate whether the created random graphs are connected:
1. we varify the reachability of all nodes with an amplitude search algorithm;
2. we verify that the Laplacian matrix associated with the graph has a null eigenvalue of multiplicity 1;
3. we verify that the irreducibility of the adjacency matrix (A) associated with the graph by solving the following inequality
