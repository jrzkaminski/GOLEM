from typing import Sequence, List, TYPE_CHECKING, Callable, Union

from golem.core.utilities.data_structures import ensure_wrapped_in_sequence

if TYPE_CHECKING:
    from golem.core.dag.graph import Graph
    from golem.core.dag.graph_node import GraphNode


def distance_to_root_level(graph: 'Graph', node: 'GraphNode') -> int:
    """Gets distance to the final output node

    Args:
        graph: graph for finding the distance
        node: search starting point

    Return:
        int: distance to root level
    """

    def child_height(parent_node: 'GraphNode') -> int:
        height = 0
        for _ in range(graph.length):
            node_children = graph.node_children(parent_node)
            if node_children:
                height += 1
                parent_node = node_children[0]
            else:
                return height

    if graph_has_cycle(graph):
        return -1
    height = child_height(node)
    return height


def distance_to_primary_level(node: 'GraphNode') -> int:
    depth = node_depth(node)
    return depth - 1 if depth > 0 else -1


def nodes_from_layer(graph: 'Graph', layer_number: int) -> Sequence['GraphNode']:
    """Gets all the nodes from the chosen layer up to the surface

    Args:
        graph: graph with nodes
        layer_number: max height of diving

    Returns:
        all nodes from the surface to the ``layer_number`` layer
    """

    def get_nodes(roots: Sequence['GraphNode'], current_height: int) -> Sequence['GraphNode']:
        """Gets all the parent nodes of ``roots``

        :param roots: nodes to get all subnodes from
        :param current_height: current diving step depth

        :return: all parent nodes of ``roots`` in one sequence:69
        """
        nodes = []
        if current_height == layer_number:
            nodes.extend(roots)
        else:
            for root in roots:
                nodes.extend(get_nodes(root.nodes_from, current_height + 1))
        return nodes

    nodes = get_nodes(graph.root_nodes(), current_height=0)
    return nodes


def ordered_subnodes_hierarchy(node: 'GraphNode') -> List['GraphNode']:
    """Gets hierarchical subnodes representation of the graph starting from the bounded node

    Returns:
        List['GraphNode']: hierarchical subnodes list starting from the bounded node
    """
    started = {node}
    visited = set()

    def subtree_impl(node):
        nodes = [node]
        for parent in node.nodes_from:
            if parent in visited:
                continue
            elif parent in started:
                raise ValueError('Can not build ordered node hierarchy: graph has cycle')
            started.add(parent)
            nodes.extend(subtree_impl(parent))
            visited.add(parent)
        return nodes

    return subtree_impl(node)


def node_depth(nodes: Union['GraphNode', Sequence['GraphNode']]) -> Union[int, List[int]]:
    """Gets the depth of the provided ``nodes`` in the graph

    Args:
        nodes: nodes to calculate the depth for

    Returns:
        int or List[int]: depth(s) of the nodes in the graph
    """
    nodes = ensure_wrapped_in_sequence(nodes)
    visited_nodes = [[node] for node in nodes]
    depth = 1
    parents = [node.nodes_from for node in nodes]
    while any(parents):
        depth += 1
        for i, ith_parents in enumerate(parents):
            grandparents = []
            for parent in ith_parents:
                if parent in visited_nodes[i]:
                    return -1
                grandparents.extend(parent.nodes_from)
            visited_nodes[i].extend(ith_parents)
            parents[i] = grandparents

    return depth


def map_dag_nodes(transform: Callable, nodes: Sequence) -> Sequence:
    """Maps nodes in dfs-order while respecting node edges.

    Args:
        transform: node transform function (maps node to node)
        nodes: sequence of nodes for mapping

    Returns:
        Sequence: sequence of transformed links with preserved relations
    """
    mapped_nodes = {}

    def map_impl(node):
        already_mapped = mapped_nodes.get(id(node))
        if already_mapped:
            return already_mapped
        # map node itself
        mapped_node = transform(node)
        # remember it to avoid recursion
        mapped_nodes[id(node)] = mapped_node
        # map its children
        mapped_node.nodes_from = list(map(map_impl, node.nodes_from))
        return mapped_node

    return list(map(map_impl, nodes))


def graph_structure(graph: 'Graph') -> str:
    """ Returns structural information about the graph - names and parameters of graph nodes.
    Represents graph info in easily readable way.

    Returns:
        str: graph structure
    """
    return '\n'.join([str(graph), *(f'{node.name} - {node.parameters}' for node in graph.nodes)])


def graph_has_cycle(graph: 'Graph') -> bool:
    """ Returns True if the graph contains a cycle and False otherwise. Implements Depth-First Search."""

    visited = {node.uid: False for node in graph.nodes}
    stack = []
    on_stack = {node.uid: False for node in graph.nodes}
    for node in graph.nodes:
        if visited[node.uid]:
            continue
        stack.append(node)
        while len(stack) > 0:
            cur_node = stack[-1]
            if not visited[cur_node.uid]:
                visited[cur_node.uid] = True
                on_stack[cur_node.uid] = True
            else:
                on_stack[cur_node.uid] = False
                stack.pop()
            for parent in cur_node.nodes_from:
                if not visited[parent.uid]:
                    stack.append(parent)
                elif on_stack[parent.uid]:
                    return True
    return False
