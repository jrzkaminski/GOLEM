from copy import deepcopy

import pytest
from hyperopt import hp

from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.search_space import SearchSpace
from golem.core.tuning.sequential import SequentialTuner
from golem.core.tuning.simultaneous import SimultaneousTuner
from test.unit.mocks.common_mocks import MockAdapter, MockObjectiveEvaluate, mock_graph_with_params, \
    opt_graph_with_params, MockNode, MockDomainStructure
from test.unit.utils import CustomMetric


def not_tunable_mock_graph():
    node_d = MockNode('d')
    node_final = MockNode('f', nodes_from=[node_d])
    graph = MockDomainStructure([node_final])

    return graph


@pytest.fixture()
def search_space():
    params_per_operation = {
        'a': {
            'a1': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 7],
                'type': 'discrete'
            },
            'a2': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-3, 1],
                'type': 'continuous'
            }
        },
        'b': {
            'b1': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["first", "second", "third"]],
                'type': 'categorical'
            },
            'b2': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            },
        },
        'e': {
            'e1': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            },
            'e2': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            }
        },
        'k': {
            'k': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1e-2, 10.0],
                'type': 'continuous'
            }
        }}
    return SearchSpace(params_per_operation)


@pytest.mark.parametrize('tuner_cls', [SimultaneousTuner, SequentialTuner, IOptTuner])
@pytest.mark.parametrize('graph, adapter, obj_eval',
                         [(mock_graph_with_params(), MockAdapter(),
                           MockObjectiveEvaluate(Objective({'random_metric': CustomMetric.get_value}))),
                          (opt_graph_with_params(), None,
                           ObjectiveEvaluate(Objective({'random_metric': CustomMetric.get_value})))])
def test_tuner_improves_metric(search_space, tuner_cls, graph, adapter, obj_eval):
    tuner = tuner_cls(obj_eval, search_space, adapter, iterations=20)
    tuned_graph = tuner.tune(deepcopy(graph))
    assert tuned_graph is not None
    assert tuner.obtained_metric is not None
    assert tuner.init_metric > tuner.obtained_metric


@pytest.mark.parametrize('tuner_cls', [SimultaneousTuner, SequentialTuner, IOptTuner])
@pytest.mark.parametrize('graph, adapter, obj_eval',
                         [(not_tunable_mock_graph(), MockAdapter(),
                           MockObjectiveEvaluate(Objective({'random_metric': CustomMetric.get_value})))])
def test_tuner_with_no_tunable_params(search_space, tuner_cls, graph, adapter, obj_eval):
    tuner = tuner_cls(obj_eval, search_space, adapter, iterations=20)
    tuned_graph = tuner.tune(deepcopy(graph))
    assert tuned_graph is not None
    assert tuner.obtained_metric is not None
    assert tuner.init_metric == tuner.obtained_metric


@pytest.mark.parametrize('graph', [mock_graph_with_params(), opt_graph_with_params(), not_tunable_mock_graph()])
def test_node_tuning(search_space, graph):
    obj_eval = MockObjectiveEvaluate(Objective({'random_metric': CustomMetric.get_value}))
    adapter = MockAdapter()
    for node_idx in range(graph.length):
        tuner = SequentialTuner(obj_eval, search_space, adapter, iterations=10)
        tuned_graph = tuner.tune_node(graph, node_idx)
        assert tuned_graph is not None
        assert tuner.obtained_metric is not None
        assert tuner.init_metric >= tuner.obtained_metric
