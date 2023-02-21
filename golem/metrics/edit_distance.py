import operator
from datetime import timedelta, datetime
from typing import Optional, Callable, Dict, Sequence

import networkx as nx
import numpy as np
import zss
from networkx import graph_edit_distance

from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.metrics.graph_metrics import min_max
from libs.netcomp import edit_distance


def _label_dist(label1: str, label2: str) -> int:
    return int(label1 != label2)


def tree_edit_dist(target_graph: nx.DiGraph, graph: nx.DiGraph,
                   with_node_substitute_dist: bool = True) -> float:
    label_dist = _label_dist if with_node_substitute_dist else lambda x, y: 1
    target_tree_root = _nx_to_zss_tree(target_graph)
    cmp_tree_root = _nx_to_zss_tree(graph)
    dist = zss.simple_distance(target_tree_root, cmp_tree_root, label_dist=label_dist)
    return dist


def _nx_to_zss_tree(graph: nx.DiGraph) -> zss.Node:
    root = _get_root_node(graph)
    tree = nx.dfs_tree(graph, source=root)
    nodes_dict = {}
    for edge in tree.edges():
        if edge[0] not in nodes_dict:
            nodes_dict[edge[0]] = zss.Node(edge[0])
        if edge[1] not in nodes_dict:
            nodes_dict[edge[1]] = zss.Node(edge[1])
        nodes_dict[edge[0]].addkid(nodes_dict[edge[1]])
    return nodes_dict[root]


def _get_root_node(nxgraph: nx.DiGraph) -> Sequence:
    source = [n for (n, d) in nxgraph.in_degree() if d == 0][0]
    return source


def get_edit_dist_metric(target_graph: nx.DiGraph,
                         timeout=timedelta(seconds=60),
                         upper_bound: Optional[int] = None,
                         requirements: Optional[GraphRequirements] = None,
                         ) -> Callable[[nx.DiGraph], float]:
    def node_match(node_content_1: Dict, node_content_2: Dict) -> bool:
        operations_do_match = node_content_1.get('name') == node_content_2.get('name')
        return True or operations_do_match

    if requirements:
        upper_bound = upper_bound or int(np.sqrt(requirements.max_depth * requirements.max_arity)),
        timeout = timeout or requirements.max_graph_fit_time

    def metric(graph: nx.DiGraph) -> float:
        ged = graph_edit_distance(target_graph, graph,
                                  node_match=node_match,
                                  upper_bound=upper_bound,
                                  timeout=timeout.total_seconds() if timeout else None,
                                 )
        return float(ged) or upper_bound

    return metric


def matrix_edit_dist(target_graph: nx.DiGraph, graph: nx.DiGraph) -> float:
    target_adj = nx.adjacency_matrix(target_graph)
    adj = nx.adjacency_matrix(graph)
    nmin, nmax = min_max(target_adj.shape[0], adj.shape[0])
    if nmin != nmax:
        shape = (nmax, nmax)
        target_adj.resize(shape)
        adj.resize(shape)
    value = edit_distance(target_adj, adj)
    return value


def try_tree_edit_distance():
    for i, n in enumerate(range(10, 200, 10)):
        g1 = nx.random_tree(n, create_using=nx.DiGraph)
        g2 = nx.random_tree(n, create_using=nx.DiGraph)

        start_time = datetime.now()
        dist = tree_edit_dist(g1, g2)
        duration = datetime.now() - start_time

        print(f'iter {i} with size={n} dist={dist}, t={duration.total_seconds():.3f}s')
        from examples.synthetic_graph_evolution.utils import draw_graphs_subplots
        draw_graphs_subplots(g1, g2)


if __name__ == "__main__":
    try_tree_edit_distance()


