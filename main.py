from functions import *


# create the array with different number of nodes on which to test our methods
k_array = [10,20,30,40,60,80,100, 150, 200, 250]

# plot running time complexity
sns_plot = complex_comparis(k_array , "er", p=0.65, sim_size=10)
# save the figure in desired format (e.g. png, pdf, svg)
plt.savefig("complex_comparis.png")

# plot space complexity
sns_plot = complex_comparis2(k_array , "er", p=0.65, sim_size=5)
plt.savefig("complex_comparis2.png")


# simulate and plot E-R graphs
sns_plot = sim_ERgraph_fast(100)
plt.savefig("ERgraph.png")


# simulate and plot R-regular graphs
sns_plot = sim_rgraph(100)
plt.savefig("Rregular.png")


# simulate and plot performances of the two topologies
sns_plot = time_plus_cost(10)
plt.savefig("TimeAndCost.png")