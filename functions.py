import networkx as nx
import numpy as np
from collections import defaultdict, deque
import random
import time
import psutil
import sys
from scipy.stats import bernoulli
from itertools import combinations
from memory_profiler import profile, memory_usage
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from topologies import FatTreeTopology, JellyfishTopology


def create_graph(k: int, typ: AnyStr = "er", p: int = 1, r: int = 1) -> nx.Graph:
    """Create a graph with the passed parameters
    k: number of nodes
    typ: type of the graph (Erdos-Renyi graph or R-regular)
    p: parameter for the Erdos-Renyi random graph
    r: parameter for the R-regular random graph
    """
    # build a graph with E-R algorithm from scratch
    if typ=="er":
        G = nx.Graph()
        G.add_nodes_from(range(k))
        # control on the probability
        if p < 0 or p > 1:
            return "Probability must be between 0 and 1"
        edges = combinations(range(k), 2)
        for e in edges:
            if bernoulli.rvs(p, size=1)[0] == 1:
                G.add_edge(*e)

    # build an r-regular graph
    elif typ=="r":
        G = nx.random_regular_graph(r, k)
    else:
        print("Not allowed type") # control on type
        return 0
    return G


def irreducibility(G: nx.Graph) -> bool:
    """Evaluate the connectivity of a graph with the irreducibility method
    """
    n = len(G.nodes)
    I = np.identity(n, dtype=int)
    A = nx.to_numpy_array(G)
    poly = [np.linalg.matrix_power(A,i) for i in range(1,n)]
    poly = np.sum(poly, axis=0) + I
    return np.all(poly > 0)

def eigen(G: nx.Graph) -> bool:
    """Evaluate the connectivity of a graph with the Laplacian matrix method
    """
    L = nx.laplacian_matrix(G).toarray()
    eigvals= np.sort(np.round(np.linalg.eigvals(L),5))
    return(eigvals[1]>0)

def bfs(G: nx.Graph) -> bool:
    """Evaluate the connectivity of a graph with the bfs method
    """
    node = random.sample(G.nodes, 1)
    queue = deque(node)
    visited = node
        
    while len(queue) != 0:
        v = queue.popleft()
        for u in G.neighbors(v):
            if u not in visited:
                visited.append(u)
                queue.append(u)
    return len(G.nodes) == len(visited)

@profile
def irreducibility2(G):
    n = len(G.nodes)
    I = np.identity(n, dtype=int)
    A = nx.to_numpy_array(G)
    poly = [np.linalg.matrix_power(A,i) for i in range(1,n)]
    poly = np.sum(poly, axis=0) + I
    return np.all(poly > 0)

@profile
def eigen2(G):
    L = nx.laplacian_matrix(G).toarray()
    eigvals= np.sort(np.round(np.linalg.eigvals(L),5))
    return(eigvals[1]>0)

@profile
def bfs2(G):
    node = random.sample(G.nodes, 1)
    queue = deque(node)
    visited = node
        
    while len(queue) != 0:
        v = queue.popleft()
        for u in G.neighbors(v):
            if u not in visited:
                visited.append(u)
                queue.append(u)
    return len(G.nodes) == len(visited)


def complex_time(k_array: list, typ: str = "er", p: int = 1, r: int = 1):
    """Evaluate complexity (in terms od time) curves as a function
    of the number of nodes K of the methods above
    """
    # create the graphs for different k
    graphs = [create_graph(k, typ, p, r) for k in k_array]
    # evaluate performance in time
    perfs = defaultdict(list)
    for g in graphs:
        # method 1
        start = time.time()
        irreducibility(g)
        end = time.time()
        perfs["irreducibility"].append(end-start)
        # method 2
        start = time.time()
        eigen(g)
        end = time.time()
        perfs["eigenvalues"].append(end-start)
        # method 3
        start = time.time()
        bfs(g)
        end = time.time()
        perfs["bfs"].append(end-start)
    return(perfs)


def complex_space(k_array: list, typ: str = "er", p: int = 1, r: int = 1):
    """Evaluate complexity (in terms od space) curves as a function
    of the number of nodes K of the methods above
    """
    # create the graphs for different k
    graphs = [create_graph(k, typ, p, r) for k in k_array]
    # evaluate performance in time
    perfs = defaultdict(list)
    for g in graphs:
        # method 1
        mem_usage = memory_usage((irreducibility2, (g, )))
        total_mem_usage = max(mem_usage) - min(mem_usage)
        perfs["irreducibility"].append(total_mem_usage)
        # method 2
        mem_usage = memory_usage((eigen2, (g, )))
        total_mem_usage = max(mem_usage) - min(mem_usage)
        perfs["eigenvalues"].append(total_mem_usage)
        # method 3
        mem_usage = memory_usage((bfs2, (g, )))
        total_mem_usage = max(mem_usage) - min(mem_usage)
        perfs["bfs"].append(total_mem_usage)
    return(perfs)


def complex_comparis(k_array: list, typ: str = "er", p: int = 1, r: int = 1, sim_size: int = 1):
    """Plots complexity (in terms od time) curves as a function
    of the number of nodes K of the methods above with a simulation
    """
    sim_data = []
    for _ in range(sim_size):
        data = complex_time(k_array, typ ,p , r)
        data1 = pd.DataFrame({"numbr nodes":k_array, "time":data["irreducibility"], "method":"irreducibility"})
        data2 = pd.DataFrame({"numbr nodes":k_array, "time":data["eigenvalues"], "method":"laplacian matrix"})
        data3 = pd.DataFrame({"numbr nodes":k_array, "time":data["bfs"], "method":"bfs"})
        sim_data.append(pd.concat([data1, data2, data3], ignore_index=True))
    data = pd.concat(sim_data, ignore_index=True)
    sns.set()
    sns.lineplot(data = data, x="numbr nodes", y="time", hue="method")
    plt.title("Time Complexity")


def complex_comparis2(k_array: list, typ: str = "er", p: int = 1, r: int = 1, sim_size: int = 1):
    """Plots complexity (in terms od space) curves as a function
    of the number of nodes K of the methods above with a simulation
    """
    sim_data = []
    for _ in range(sim_size):
        data = complex_space(k_array, typ ,p, r)
        data1 = pd.DataFrame({"numbr nodes":k_array, "time":data["irreducibility"], "method":"irreducibility"})
        data2 = pd.DataFrame({"numbr nodes":k_array, "time":data["eigenvalues"], "method":"laplacian matrix"})
        data3 = pd.DataFrame({"numbr nodes":k_array, "time":data["bfs"], "method":"bfs"})
        sim_data.append(pd.concat([data1, data2, data3], ignore_index=True))
    data = pd.concat(sim_data, ignore_index=True)
    sns.set()
    sns.lineplot(data = data, x="numbr nodes", y="time", hue="method")
    plt.title("Space Complexity")


def sim_ERgraph(size: int):
    """Plots the probability a Erdos-Renyi graph of size K=100
    is connected (we'll call it connectivity rate) for different values of p
    """
    probs = list(np.linspace(0,1, 50))
    k = 100
    conn = np.array(list(map(lambda p: [bfs(create_graph(k, typ="er", p=p)) for _ in range(size)], probs)))
    data = pd.DataFrame({"p": probs, "connectivity_rate": np.mean(conn, axis=1)})
    # Find the index of the first occurrence where the connectivity rate reaches 1
    first_index = np.argmax(data["connectivity rate"] == 1)
    # Get the corresponding probability value
    first_prob = data.iloc[first_index]["p"]
    sns.set()
    sns.lineplot(data=data, x="p", y="connectivity rate", color="darkviolet")
    plt.vlines(x=first_prob, ymin=0, ymax=1, color='violet', linestyle='--')
    plt.suptitle("ER-graph connectivity rate", fontsize=16)
    plt.title(f"Simulation size: {size}", fontsize=12, color="gray")


def sim_rgraph(size):
    """Plots the probability a r-regular random graph with r=2 and r=8
    is connected (we'll call it connectivity rate) for different values of K<=100
    """
    k_list = list(range(10, 100, 5))
    conn_r2 = np.array(list(map(lambda k: [bfs(create_graph(k, typ="r", r=2)) for _ in range(size)], k_list)))
    conn_r8 = np.array(list(map(lambda k: [nx.is_connected(create_graph(k, typ="r", r=8)) for _ in range(size)], k_list)))

    data = pd.DataFrame({"numbr nodes": k_list, "connectivity rate": np.mean(conn_r2, axis=1), "r": "r = 2"})
    data2 = pd.DataFrame({"numbr nodes": k_list, "connectivity rate": np.mean(conn_r8, axis=1), "r": "r = 8"})
    data = pd.concat([data, data2], ignore_index=True)

    # Plot the data
    sns.set()
    palette = {"r = 2": "darkorange", "r = 8": "blue"}
    sns.lineplot(data=data, x="numbr nodes", y="connectivity rate", hue="r", palette=palette)
    plt.legend(loc="center right", title="R-Value")
    plt.suptitle("R-regular graph connectivity rate", fontsize=16)
    plt.title(f"Simulation size: {size}", fontsize=12, color="gray")


# implementation with fast generation of the E-R graph
def sim_ERgraph_fast(size: int):
    """Plots the probability a Erdos-Renyi graph of size K=100
    is connected (we'll call it connectivity rate) for different values of p
    """
    probs = list(np.linspace(0,1, 50))
    k = 100
    conn = np.array(list(map(lambda p: [bfs(nx.fast_gnp_random_graph(k, p=p)) for _ in range(size)], probs)))
    data = pd.DataFrame({"p": probs, "connectivity rate": np.mean(conn, axis=1)})
    # Find the index of the first occurrence where the connectivity rate reaches 1
    first_index = np.argmax(data["connectivity rate"] == 1)
    # Get the corresponding probability value
    first_prob = data.iloc[first_index]["p"]
    sns.set()
    sns.lineplot(data=data, x="p", y="connectivity rate", color="darkviolet")
    plt.vlines(x=first_prob, ymin=0, ymax=1, color='violet', linestyle='--')
    plt.suptitle("ER-graph connectivity rate", fontsize=16)
    plt.title(f"Simulation size: {size}", fontsize=12, color="gray")


def simulate_topology(M: int, topology: str, N):
    """ Monte Carlo simulation to get the Mean Response Time and the Mean Utilization Time
    of a given topology for many different sizes (range [2, 10000])
    M: simulation size
    topology: the used topology
    N: array of topology sizes
    """

    #N = np.arange(0, 10001, 100)
    #N[0] = 2
    # link capacity
    C = 10 * 0.000125 ## 1 Gbit = 0.000125TB
    tau = 5 * 1e-6 ## 1 micro-sec = 1e-6 seconds
    # input & output file size
    L_f = 4
    L_o = 4
    # setup time 
    T_o = 30
    # TCP handshake delay
    f = 48/1500
    E_X = 8 * 3600

    # if the topology is a Fat-tree it is deterministic and we can avoid to create it at each iteration of the simulation
    if topology=="fat":  
        topology = FatTreeTopology(64)
    
    # init the results
    ER = np.zeros((M,len(N)))
    ETheta = np.zeros((M,len(N)))

    for m in tqdm(range(M)):
        R = []
        Theta = []

        # if the topology is a Jellyfish its creation is completely random
        if topology=="jelly":
            topology = JellyfishTopology(s=2048,n=64,r=32)

        # single step of the simulation
        source_server = random.choice(topology.get_servers()) # sample a server A
        target_nodes = list(topology.get_servers()) # init the target nodes for the shortest path
        neighbours = dict()
        for server in  target_nodes: # for every server
           a = nx.shortest_path_length(topology.graph, source=source_server,target = server) # compute he shortest path from source
           neighbours[server] = a  
        neighbours = {k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1])} # sort the dict

        for n in N:
            # get the N closest neighbours
            neigh = dict(list(neighbours.items())[1:(n+1)])
            h_i = np.array(list(neigh.values()))
            # round trip time for each node 
            RTT_i = 2*tau*h_i 
            # avg throughput for each neighbour
            denom = sum(1/RTT_i) # normalizing constant
            phi_i = (C*(1/RTT_i))/denom
            # forth transmission time
            forth_i = np.divide(L_f/n, phi_i) * (1+f)
            # back transmission time
            L_o_i = np.random.uniform(low=0.0, high=(2*L_o)/n, size=n)
            back_i = np.divide(L_o_i, phi_i) * (1+f)
            # computing time on nodes
            X_i = np.random.exponential(scale=E_X/n, size=n)
            # server usage time
            Theta.append(sum(T_o + X_i))
            # rensponse time 
            R.append(max(forth_i + back_i + T_o + X_i)) # take max for the Straggler problem
        ER[m] = np.array(R)
        ETheta[m] = np.array(Theta)
    
    E_R = np.mean(ER, axis = 0)
    E_Theta = np.mean(ETheta, axis = 0)
    return (E_R, E_Theta)


def time_plus_cost(M):
    """Plots response time and utilization cost respectively"""

    eps = 0.1

    # strucure the graanulrity of the plots
    N = np.arange(1, 1500, 10)
    critical_points = np.array([32, 1024])
    N2 = np.arange(1500, 3000, 200)
    N3 = np.arange(3000, 10001, 500)
    N = np.sort(np.concatenate((N, critical_points, N2, N3)))

    # setup time 
    T_o = 30
    E_X = 8 * 3600
    R_baseline = T_o + E_X
    S_baseline = R_baseline + (R_baseline*eps)

    # evaluate topology response time and utilization time
    E_R_jelly, E_theta_jelly = simulate_topology(M, "jelly",N)
    E_R_fat, E_theta_fat = simulate_topology(M, "fat",N)


    # evaluate the topology cost
    jelly_cost = E_R_jelly + (eps * E_theta_jelly)
    fat_cost =  E_R_fat + (eps * E_theta_fat)


    # nomralize for the baseline
    jelly_cost = jelly_cost/S_baseline
    fat_cost = fat_cost/S_baseline

    # nomralize for the baseline
    E_R_jelly = E_R_jelly/R_baseline
    E_R_fat = E_R_fat/R_baseline


    data_R = pd.DataFrame({"topology": "Fat-tree", "E[R]": E_R_fat, "N":N})
    tmp = pd.DataFrame({"topology": "Jellyfish", "E[R]": E_R_jelly, "N":N})
    data_R = pd.concat([data_R, tmp], ignore_index=True)

    data_C = pd.DataFrame({"topology": "Fat-tree", "S": fat_cost, "N":N})
    tmp = pd.DataFrame({"topology": "Jellyfish", "S": jelly_cost, "N":N})
    data_C = pd.concat([data_C, tmp], ignore_index=True)

    sns.set()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    palette = {"Fat-tree": "darkviolet", "Jellyfish": "chocolate"}

    # Add horizontal lines for the baseline values
    ax1.axhline(1, color='lightblue', linestyle='--')
    ax2.axhline(1, color='lightblue', linestyle='--')

    # Find the min value in the data_R DataFrame and add a vertical line
    min_idx_R = data_R['E[R]'].idxmin()
    min_val_R = data_R.loc[min_idx_R, 'N']
    ax1.axvline(min_val_R, color='gray', linestyle='--')

    # Find the min value in the data_C DataFrame and add a vertical line
    min_idx_C = data_C['S'].idxmin()
    min_val_C = data_C.loc[min_idx_C, 'N']
    ax2.axvline(min_val_C, color='gray', linestyle='--')

    sns.lineplot(data = data_R, x="N", y="E[R]", hue="topology", ax=ax1, palette=palette)
    sns.lineplot(data = data_C, x="N", y="S", hue="topology", ax=ax2, palette=palette)
    ax1.set_title("Mean response time", fontsize=16)
    ax1.text(0.5, -0.15, f"Simulation size: {M} - Scale: logarithmic", fontsize=12, color="gray", ha='center', va='bottom', transform=ax1.transAxes)
    ax2.set_title("Job running cost", fontsize=16)
    ax2.text(0.5, -0.15, f"Simulation size: {M}", fontsize=12, color="gray", ha='center', va='bottom', transform=ax2.transAxes)

    # Add text to highlight and explain the min values and baseline
    ax1.text(min_val_R+800, 0.33, f"Min Value: {min_val_R}", ha='center', fontsize=12, color='gray')
    ax1.text(8000, 0.95, "Baseline", ha='center', fontsize=14, color='lightblue')

    ax2.text(min_val_C +800, 0.4, f"Min Value: {min_val_C}", ha='center', fontsize=12, color='gray')
    ax2.text(2000, 0.95, "Baseline", ha='center', fontsize=14, color='lightblue')

    # Add the zoom lens effect around the min value of the second plot
    ax2_inset = inset_axes(ax2, width="40%", height="45%", loc="lower right", borderpad=2)
    new_dat1 = data_C[data_C["topology"] == "Jellyfish"]
    new_dat2 = data_C[data_C["topology"] == "Fat-tree"]
    ax2_inset.plot(new_dat1["N"], new_dat1["S"], color= palette["Jellyfish"])
    ax2_inset.plot(new_dat2["N"], new_dat2["S"], color= palette["Fat-tree"])
    ax2_inset.axvline(min_val_C, color='gray', linestyle='--')
    ax2_inset.set_xlim(min_val_C - 500, min_val_C + 1300)  # Set the x-axis limits for the zoomed region
    min_val_C_y = data_C.loc[min_idx_C, 'S']  # Get the y value at the min value of the second plot
    ax2_inset.set_ylim(min_val_C_y -0.04, min_val_C_y + 0.27)  # Set the y-axis limits for the zoomed region

    # Add a rectangle to indicate the zoomed region in the main plot
    rect = plt.Rectangle((min_val_C - 1000, min_val_C_y - 0.1), 2000, 0.2, linewidth=1, edgecolor='violet', facecolor='none', linestyle='--')
    ax2.add_patch(rect)

    # Adjust legends
    ax1.legend(fontsize="large", loc="upper right", title_fontsize='x-large', title='Topology')
    ax2. legend(fontsize="large", loc="upper center", title_fontsize='x-large', title='Topology')

    # Adjust the space between subplots
    plt.subplots_adjust(wspace=0.1)
    plt.show()

