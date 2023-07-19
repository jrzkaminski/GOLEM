import pytest

from golem.core.adapter import DirectAdapter
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.elitism import Elitism, ElitismTypesEnum
from golem.core.optimisers.opt_history_objects.individual import Individual
from test.unit.optimizers.gp_operators.test_selection import get_objective
from test.unit.utils import graph_first, graph_second, graph_third, graph_fourth, graph_fifth


@pytest.fixture()
def set_up():
    adapter = DirectAdapter()
    graphs = [graph_first(), graph_second(), graph_third(), graph_fourth(), graph_fifth()]
    population = [Individual(adapter.adapt(graph)) for graph in graphs]
    for ind in population:
        ind.set_evaluation_result(get_objective(ind.graph))
    population, best_individuals = population[:4], population[2:]

    return best_individuals, population


def test_keep_n_best_elitism(set_up):
    best_individuals, population = set_up
    elitism = Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.keep_n_best))
    new_population = elitism(best_individuals, population)
    for best_ind in best_individuals:
        # checks that new population contains the best individuals and `keep_n_best_elitism` does not duplicate it
        assert new_population.count(best_ind) == 1
    assert len(population) == len(new_population)


def test_replace_worst(set_up):
    best_individuals, population = set_up
    elitism = Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.replace_worst))
    new_population = elitism(best_individuals, population)
    for ind in population:
        if ind not in new_population:
            assert all(ind.fitness <= best_ind.fitness for best_ind in new_population)
    assert len(new_population) == len(population)


def test_elitism_not_applicable(set_up):
    best_individuals, population = set_up
    elitisms = [
        Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.replace_worst,
                                      multi_objective=True)),
        Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.replace_worst,
                                      pop_size=4, min_pop_size_with_elitism=5)),
        Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.none)),
    ]
    for elitism in elitisms:
        new_population = elitism(best_individuals, population)
        assert new_population == population
