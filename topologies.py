import networkx as nx
import random
from tqdm import tqdm
from matplotlib.pyplot import plt
from functions import *

import networkx as nx
import random
from tqdm import tqdm

class FatTreeTopology:
    def __init__(self, n):
        self.n = n
        self.graph = self.create_topology()
        self.servers = [node for node in self.graph.nodes if node.startswith("server")]
        self.switches = [node for node in self.graph.nodes if not node.startswith("switch")]

    def create_topology(self):
        num_core_switches = (self.n // 2) ** 2   #
        num_agg_switches = num_edge_switches = self.n * (self.n // 2)  #

        num_switches = num_core_switches + num_agg_switches + num_edge_switches
        num_servers = num_edge_switches * (self.n // 2)

        # Create the graph object
        graph = nx.Graph()

        # Add switches and servers to the graph
        for i in range(num_switches):
            if i < num_core_switches:
                graph.add_node(f"core_{i}", level=0)
            elif i < num_core_switches + num_agg_switches:
                graph.add_node(f"agg_{i - num_core_switches}", level=1)
            else:
                graph.add_node(f"edge_{i - num_core_switches - num_agg_switches}", level=2)

        for i in range(num_servers):
            graph.add_node(f"server_{i}")
        

        # Add linns between switches and servers
        for i in range(num_edge_switches):   ## for each edge switch
            for j in range(self.n // 2):     ## for each port of the edge switch
                server_id = i * (self.n // 2) + j
                graph.add_edge(f"edge_{i}", f"server_{server_id}")



        cost = 0
        for i in range(num_agg_switches):
            if (i % (self.n//2)) == 0 and (i != 0):
                cost = 0  
            for j in range(self.n//2):
                core_id = j + cost
                graph.add_edge(f"core_{core_id}", f"agg_{i}")
            cost += self.n//2
  

        cost = 0
        # Add links between aggregate and edge switches
        for i in range(num_agg_switches):
            if (i % (self.n // 2)) == 0 and (i!= 0):
                cost += self.n // 2
            for j in range(self.n // 2):
                edge_id = j + cost 
                graph.add_edge(f"agg_{i}", f"edge_{edge_id}")

        return graph

    def get_servers(self):
        return self.servers

    def get_switches(self):
        return self.switches

    def __str__(self):
        return f"Fat-Tree topology (n={self.n})"


    def get_closest_nodes(self, server_node, N):
        """
        Returns the top N closest neighboring nodes to a specific server node in terms of number of hops.
        """
        # Check if server_node is a valid server node in the topology
        if server_node not in self.get_servers():
            raise ValueError("Invalid server node provided")
        
        shortest_paths = {}
        
        # Get the shortest paths from the server node to all other nodes in the topology
        for server in self.get_servers():
            shortest_paths[server] = nx.shortest_path_length(self.graph, source=server_node, target = server)
        

        closest_nodes = dict(sorted(shortest_paths.items(), key=lambda x:x[1]))
        # Sort the nodes by shortest path length and return the top N closest neighboring nodes
        closest_nodes = dict(list(closest_nodes.items())[1:(N+1)])
        return closest_nodes
    
    def draw(self):
        # Create a dictionary to map node types to colors
        node_colors = {
            "core": "red",
            "agg": "blue",
            "edge": "green",
            "server": "purple"
        }

        # Create a list of node colors for each node in the graph
        colors = [node_colors[node.split("_")[0]] for node in self.graph.nodes]

        # Set the positions of the nodes
        pos = nx.spring_layout(self.graph, seed=42)

        # Draw the graph with specified node colors and positions
        nx.draw(self.graph, pos, node_color=colors, with_labels=True)

        # Create a legend for the node colors
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", label="Core Switch", markersize=10, markerfacecolor="red"),
            plt.Line2D([0], [0], marker="o", color="w", label="Aggregate Switch", markersize=10, markerfacecolor="blue"),
            plt.Line2D([0], [0], marker="o", color="w", label="Edge Switch", markersize=10, markerfacecolor="green"),
            plt.Line2D([0], [0], marker="o", color="w", label="Server", markersize=10, markerfacecolor="purple")
        ]

        plt.legend(handles=legend_elements, loc="upper left")

        # Display the graph
        plt.show()


    
# Jellyfish implementation from scratch but it is really slow
class SlowJellyfishTopology:
    def __init__(self, s, n, r):
        self.s = s  # number of switches
        self.n = n  # number of ports per switch
        self.r = r  # number of ports per switch connected to other switches
        self.graph = self.create_topology()

    def create_topology(self):
        # Create the graph object
        graph = nx.Graph()

        # Create a list to keep track of the free ports on each switch
        free_ports = {f"switch_{i}": self.n for i in range(self.s)}

        # Create a list of all switch nodes
        switches = [f"switch_{i}" for i in range(self.s)]

        # Connect switches randomly until no further links can be added
        while True:
            # Get a list of switch pairs with free ports and not already connected
            switch_pairs = [(i, j) for i in switches for j in switches
                            if (i != j) and (free_ports[i] > self.r) and (free_ports[j] > self.r) and (not graph.has_edge(i, j))]
            

            # If there are no switch pairs available, exit the loop
            if not switch_pairs:
                break

            # Choose a random switch pair
            i, j = random.choice(switch_pairs)

            # Connect the switches
            graph.add_edge(i, j)

            # Reduce the number of free ports on the connected switches
            free_ports[i] -= 1
            free_ports[j] -= 1

        while True:
            count = 0
            for v in tqdm(free_ports):
                switch_links = [(i, j) for i in switches for j in switches if (i != j) and graph.has_edge(i, j)]

                if free_ports[v] < 2:
                    count += 1

                elif free_ports[v] >= 2:
                    
                    i, j = random.choice(switch_links)
                    while (i == v or j== v ):
                        i, j = random.choice(switch_links)
            
                    graph.remove_edge(i,j)
                    
                    graph.add_edge(v,i)
                    graph.add_edge(v,j)

                    free_ports[v] -= 2

            if count == len(free_ports):
                print('sono qui')
                break

        # Add servers to the graph
        for i in range(self.s):
            switch = f"switch_{i}"
            for j in range(self.n - self.r):
                server = f"{switch}:{j}"
                graph.add_node(server)
                graph.add_edge(switch, server)

        return graph
    

    def get_closest_nodes(self, server_node, N):
        """Returns the top N closest neighboring nodes to a specific server node in terms of number of hops.
        """
        # Check if server_node is a valid server node in the topology
        if server_node not in self.get_servers():
            raise ValueError("Invalid server node provided")
        
        shortest_paths = {}
        
        # Get the shortest paths from the server node to all other nodes in the topology
        for server in self.get_servers():
            shortest_paths[server] = nx.shortest_path_length(self.graph, source=server_node, target = server)
        

        closest_nodes = dict(sorted(shortest_paths.items(), key=lambda x:x[1]))
        # Sort the nodes by shortest path length and return the top N closest neighboring nodes
        closest_nodes = dict(list(closest_nodes.items())[1:(N+1)])
        return closest_nodes


# fast implementation if the Jellyfish with r-regular random graph
class JellyfishTopology:
    def __init__(self, s, n, r):
        self.n = n  # number of ports per switch
        self.s = s  # number of switches
        self.r = r  # number of ports per switch connected to other switches
        self.graph = self.create_topology()



    def create_topology(self):
        graph = create_graph(k=self.s, typ="r",r=self.r)
        labels = []
        nx.set_node_attributes(graph, labels, "labels")
        labels.append("switch")
        switches = list(graph.nodes())
        for i in range(len(switches)):
            for j in range(self.n-self.r):
                graph.add_node(f"server{i}_{j}")
                graph.add_edge(switches[i], f'server{i}_{j}')
        return graph

    def get_servers(self):
        return [i for i in list(self.graph.nodes()) if not isinstance(i, int)]

                
    def get_closest_nodes(self, server_node, N):
        """
        Returns the top N closest neighboring nodes to a specific server node in terms of number of hops.
        """
        # Check if server_node is a valid server node in the topology
        if server_node not in self.get_servers():
            raise ValueError("Invalid server node provided")
        
        shortest_paths = {}
        
        # Get the shortest paths from the server node to all other nodes in the topology
        for server in self.get_servers():
            shortest_paths[server] = nx.shortest_path_length(self.graph, source=server_node, target = server)
        

        closest_nodes = dict(sorted(shortest_paths.items(), key=lambda x:x[1]))
        # Sort the nodes by shortest path length and return the top N closest neighboring nodes
        closest_nodes = dict(list(closest_nodes.items())[1:(N+1)])
        return closest_nodes
    
    def draw(self):
        # Create a dictionary to map node types to colors
        node_colors = {
            "switch": "blue",
            "server": "purple"
        }

        # Create a list of node colors for each node in the graph
        colors = [node_colors["switch"] if isinstance(node, int) else node_colors["server"] for node in list(self.graph.nodes)]

        # Set the positions of the nodes
        pos = nx.spring_layout(self.graph, seed=42)

        # Draw the graph with specified node colors and positions
        nx.draw(self.graph, pos, node_color=colors, with_labels=True)

        # Create a legend for the node colors
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", label="Switch", markersize=10, markerfacecolor="blue"),
            plt.Line2D([0], [0], marker="o", color="w", label="Server", markersize=10, markerfacecolor="purple")
        ]

        plt.legend(handles=legend_elements, loc="upper left")

        # Display the graph
        plt.show()
