import numpy as np
import networkx as nx
import json
import os



DATA_DIR = "/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/invest_agent_data/graph_data"


def generate_investor_network_dag(N=100, p_inst=0.05, k_out_mean=10):
    """Generate a directed acyclic graph (DAG) representing an investor network.
    Nodes represent investors of two types: 'retail' and 'institutional'.
    Edges represent investment relationships with weights indicating investment amounts.
    The DAG constraint ensures there are no directed cycles.
    """
    # 1. assign types
    n_inst = max(1, int(N * p_inst))  # number of institutional investors (at least 1)
    types = np.array(["inst"] * n_inst + ["retail"] * (N - n_inst))
    np.random.shuffle(types)

    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, type=types[i])

    # 2. sample node attributes according to type
    for i in range(N):
        if types[i] == "retail":
            alpha = np.random.gamma(shape=2.0, scale=1.0)  # risk aversion coefficient
            lam = max(1.0, np.random.normal(2.25, 0.5))  # loss aversion coefficient
            m = 1.0 + np.random.lognormal(mean=0.2, sigma=0.25)  # overconfidence coefficient
        else:  # institutional
            alpha = np.random.gamma(shape=1.2, scale=0.6)
            lam = max(1.0, np.random.normal(1.8, 0.4))
            m = 1.0 + np.random.lognormal(mean=0.1, sigma=0.12)
        # convert to native float for GraphML compatibility
        G.nodes[i].update({"alpha": float(alpha), "lambda": float(lam), "m": float(m)})

    # 3. create edges
    # For each node, sample out-degree from Poisson(k_out_mean)
    # Only allow edges to nodes with higher index to ensure DAG
    in_deg = np.ones(N)  # initial small attractiveness
    for i in range(N):
        k = np.random.poisson(lam=k_out_mean)
        candidates = np.arange(i + 1, N)
        if len(candidates) == 0:
            continue

        # Type-based edge probability (retail prefers inst, inst -> retail less likely)
        type_bias = np.ones_like(candidates, dtype=float)
        for idx, j in enumerate(candidates):
            if types[i] == "retail" and types[j] == "inst":
                type_bias[idx] = 3.0
            elif types[i] == "retail" and types[j] == "retail":
                type_bias[idx] = 0.5
            elif types[i] == "inst" and types[j] == "inst":
                type_bias[idx] = 1.5
            elif types[i] == "inst" and types[j] == "retail":
                type_bias[idx] = 0.3

        probs = in_deg[candidates] * type_bias
        probs = probs / probs.sum()
        chosen = np.random.choice(candidates, size=min(k, len(candidates)), replace=False, p=probs)

        # add edges with weight according to type pair
        for j in chosen:
            pair = (types[i], types[j])
            if pair == ("retail", "inst"):
                w = np.random.gamma(2.0, 1.5)
            elif pair == ("retail", "retail"):
                w = np.random.gamma(1.0, 0.5)
            elif pair == ("inst", "inst"):
                w = np.random.gamma(1.2, 0.8)
            else:  # inst -> retail
                w = np.random.gamma(0.2, 0.25)
            G.add_edge(i, j, weight=float(w))
            in_deg[j] += 1.0
    return G


def convert_attrs_to_native(G):
    """Convert all numpy scalar attributes to native Python types.

    Parameters:
    G (nx.DiGraph): Input graph with possible numpy scalar attributes.

    Returns:
    nx.DiGraph: Graph with all attributes converted to native Python types.
    """
    for n, attrs in G.nodes(data=True):
        for k, v in attrs.items():
            if isinstance(v, np.generic):
                attrs[k] = v.item()
    for u, v, attrs in G.edges(data=True):
        for k, value in attrs.items():
            if isinstance(value, np.generic):
                attrs[k] = value.item()
    return G


def generate_node_list(G):
    """Generate a bottom-up node list with normalized outgoing edge weights.
    Each node item contains:
    - 'id': node ID
    - 'type', 'alpha', 'lambda', 'm': node features
    - 'targets': list of dicts {'id': target_id, 'weight': normalized_weight}

    Parameters:
    G (nx.DiGraph): Input directed acyclic graph.

    Returns:
    list: List of node entries in bottom-up order.
    """
    topo_order = list(nx.topological_sort(G))
    topo_order.reverse()  # bottom-up order

    node_list = []
    for node in topo_order:
        data = G.nodes[node]
        # collect out edges
        out_edges = [(t, G.edges[node, t]["weight"]) for t in G.successors(node)]
        # normalize weights
        total_w = sum(w for _, w in out_edges)
        targets = [{"id": int(t), "weight": float(w / total_w if total_w > 0 else 0.0)} for t, w in out_edges]

        node_entry = {
            "id": int(node),
            "type": str(data["type"]),
            "alpha": float(data["alpha"]),
            "lambda": float(data["lambda"]),
            "m": float(data["m"]),
            "targets": targets,
        }
        node_list.append(node_entry)
    return node_list


if __name__ == "__main__":
    # Generate investor network DAG
    G = generate_investor_network_dag(N=5, p_inst=0.05, k_out_mean=3)
    G_clean = convert_attrs_to_native(G)

    # Save to GraphML (for visualization)
    nx.write_graphml(G_clean, os.path.join(DATA_DIR, "investor_network.graphml"))
    print("Graph saved to investor_network.graphml")

    # Generate bottom-up node list with normalized edge weights
    node_list = generate_node_list(G_clean)
    with open(os.path.join(DATA_DIR, "investor_node_list.json"), "w") as f:
        json.dump(node_list, f, indent=2)

    # Load JSON example
    with open(os.path.join(DATA_DIR, "investor_node_list.json")) as f:
        loaded_node_list = json.load(f)
    print("Loaded", len(loaded_node_list), "nodes.")
    print("First node entry:", loaded_node_list[0])
    