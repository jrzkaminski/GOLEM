from datetime import timedelta
from functools import partial
from typing import Type, Optional, Sequence

import networkx as nx

from examples.synthetic_graph_evolution.experiment_setup import run_experiments
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.verification_rules import has_no_self_cycled_nodes
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.meta.surrogate_model import SurrogateModel, RandomValuesSurrogateModel
from golem.core.optimisers.meta.surrogate_optimizer import SurrogateEachNgenOptimizer
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.metrics.graph_metrics import spectral_dist


def surrogate_graph_search_setup(target_graph: nx.DiGraph,
                                 optimizer_cls: Type[GraphOptimizer] = SurrogateEachNgenOptimizer,
                                 surrogate_model: Type[SurrogateModel] = RandomValuesSurrogateModel(),
                                 node_types: Sequence[str] = ('X',),
                                 timeout: Optional[timedelta] = None,
                                 num_iterations: Optional[int] = None):
    # Setup parameters
    num_nodes = target_graph.number_of_nodes()
    requirements = GraphRequirements(
        max_arity=num_nodes,
        max_depth=num_nodes,
        early_stopping_timeout=5,
        early_stopping_iterations=1000,
        keep_n_best=4,
        timeout=timeout,
        num_of_generations=num_iterations,
        n_jobs=-1,
        history_dir=None,
    )
    gp_params = GPAlgorithmParameters(
        multi_objective=True,
        genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
        mutation_types=[
            MutationTypesEnum.single_add,
            MutationTypesEnum.single_drop,
            MutationTypesEnum.single_edge,
        ]
    )
    graph_gen_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),
        rules_for_constraint=[has_no_self_cycled_nodes],
        available_node_types=node_types,
    )

    # Setup objective that measures some graph-theoretic similarity measure
    objective = Objective(
        quality_metrics={
            'sp_adj': partial(spectral_dist, target_graph, kind='adjacency')
        }
    )

    # Generate simple initial population with line graphs
    initial_graphs = [generate_labeled_graph('line', k + 3)
                      for k in range(gp_params.pop_size)]
    initial_graphs = graph_gen_params.adapter.adapt(initial_graphs)

    # Build the optimizer
    optimiser = optimizer_cls(objective, initial_graphs, requirements, graph_gen_params, gp_params,
                              surrogate_model=surrogate_model)
    return optimiser, objective


if __name__ == '__main__':
    results_log = run_experiments(
        optimizer_setup=partial(surrogate_graph_search_setup,
                                surrogate_model=RandomValuesSurrogateModel()),
        optimizer_cls=SurrogateEachNgenOptimizer,
        graph_names=['2ring', 'gnp'],
        graph_sizes=[30, 100],
        num_trials=1,
        trial_timeout=5,
        trial_iterations=2000,
        visualize=True)
    print(results_log)
