import itertools as it

import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()
edges = [(0, 1), (0, 2), (0, 3), (2, 3)]
ret_edges = []
for e in edges:
    ret_edges.append(e[::-1])
edges = edges + ret_edges
G.add_edges_from(edges)
pos = nx.spring_layout(G)
connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
_, axs = plt.subplots(1, 3, tight_layout=True, figsize=(10, 4))
axs = axs.flatten()
node_color = ["blue"] * len(G)
node_color[2] = "red"
node_color[0] = "orange"
node_color[3] = "orange"
nx.draw(
    G,
    connectionstyle=connectionstyle,
    arrowsize=20,
    node_color=node_color,
    with_labels=True,
    pos=pos,
    ax=axs[0],
)
axs[0].set_title("t=0")
node_color = ["blue"] * len(G)
node_color[2] = "orange"
node_color[0] = "orange"
node_color[3] = "red"
nx.draw(
    G,
    connectionstyle=connectionstyle,
    arrowsize=20,
    node_color=node_color,
    with_labels=True,
    pos=pos,
    ax=axs[1],
)
axs[1].set_title("t=1")
node_color = ["blue"] * len(G)
node_color[2] = "orange"
node_color[0] = "red"
node_color[3] = "orange"
node_color[1] = "orange"
nx.draw(
    G,
    connectionstyle=connectionstyle,
    arrowsize=20,
    node_color=node_color,
    with_labels=True,
    pos=pos,
    ax=axs[2],
)
axs[2].set_title("t=2")
plt.savefig("./results/propagation_example.png", dpi=150, bbox_inches="tight")
