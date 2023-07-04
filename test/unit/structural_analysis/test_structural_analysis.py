from golem.core.optimisers.graph import OptNode, OptGraph
from golem.structural_analysis.graph_sa.graph_structural_analysis import GraphStructuralAnalysis


def get_graph_with_duplicate_operations():
    node_scaling = OptNode('scaling')
    node_pf = OptNode('poly_features', nodes_from=[node_scaling])
    node_scaling2 = OptNode('scaling', nodes_from=[node_pf])
    node_scaling3 = OptNode('scaling', nodes_from=[node_scaling2])
    node_scaling4 = OptNode('scaling', nodes_from=[node_scaling3])
    node_rf = OptNode('rf', nodes_from=[node_scaling4])
    pipeline = OptGraph(node_rf)
    return pipeline


def get_result_pipeline_after_removing_duplicates():
    node_scaling = OptNode('scaling')
    node_pf = OptNode('poly_features', nodes_from=[node_scaling])
    node_scaling2 = OptNode('scaling', nodes_from=[node_pf])
    node_rf = OptNode('rf', nodes_from=[node_scaling2])
    pipeline = OptGraph(node_rf)
    return pipeline


def test_pipeline_preprocessing():
    """ Checks whether duplicates of consecutive nodes are removed correctly during pipeline preprocessing """
    graph = get_graph_with_duplicate_operations()

    graph = GraphStructuralAnalysis.graph_preprocessing(graph=graph)

    res_graph = get_result_pipeline_after_removing_duplicates()

    assert graph == res_graph
