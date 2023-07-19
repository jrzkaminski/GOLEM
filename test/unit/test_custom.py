import random

import numpy as np
from golem.core.adapter import DirectAdapter
from golem.core.dag.convert import graph_structure_as_nx_graph
from golem.core.dag.graph_delegate import GraphDelegate
from golem.core.dag.linked_graph_node import LinkedGraphNode
from golem.core.dag.verification_rules import has_no_self_cycled_nodes
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.regularization import RegularizationTypesEnum
from golem.core.optimisers.initial_graphs_generator import InitialPopulationGenerator
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.objective.objective_eval import ObjectiveEvaluate
from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from test.unit.utils import graph_fifth, graph_fourth, graph_third, graph_second, graph_first

random.seed(1)
np.random.seed(1)


class CustomModel(GraphDelegate):

    def evaluate(self):
        return 0


class CustomNode(LinkedGraphNode):
    def __str__(self):
        return f'custom_{str(self.name)}'


def custom_metric(custom_model: CustomModel):
    _, labels = graph_structure_as_nx_graph(custom_model)

    return -len(labels) + custom_model.evaluate()


def test_custom_graph_opt():
    """Test checks for the use case of custom graph optimisation:
    that it can be initialised without problem and returns sane result."""

    nodes_types = ['A', 'B', 'C', 'D']
    rules = [has_no_self_cycled_nodes]

    requirements = GraphRequirements(
        num_of_generations=5,
        show_progress=False)

    optimiser_parameters = GPAlgorithmParameters(
        pop_size=5,
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[
            MutationTypesEnum.simple,
            MutationTypesEnum.reduce,
            MutationTypesEnum.growth,
            MutationTypesEnum.local_growth],
        regularization_type=RegularizationTypesEnum.none)

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(CustomModel, CustomNode),
        rules_for_constraint=rules,
        node_factory=DefaultOptNodeFactory(available_node_types=nodes_types))

    objective = Objective({'custom': custom_metric})
    initial_graphs = [graph_first(), graph_second(), graph_third(), graph_fourth(), graph_fifth()]
    init_population = InitialPopulationGenerator(optimiser_parameters.pop_size,
                                                 graph_generation_params, requirements)\
        .with_initial_graphs(initial_graphs)()
    optimiser = EvoGraphOptimizer(
        graph_generation_params=graph_generation_params,
        objective=objective,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements,
        initial_graphs=init_population)

    objective_eval = ObjectiveEvaluate(objective)
    optimized_graphs = optimiser.optimise(objective_eval)
    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graphs[0])

    assert optimized_network is not None
    assert isinstance(optimized_network, CustomModel)
    assert isinstance(optimized_network.nodes[0], CustomNode)
    assert optimized_network.length > 1
