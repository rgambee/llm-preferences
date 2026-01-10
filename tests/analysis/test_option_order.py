import numpy as np

from llmprefs.analysis.option_order import (
    NUM_OPTION_ORDERINGS,
    NUM_POSSIBLE_OUTCOMES,
    analyze_option_order,
    compile_observations,
)
from llmprefs.testing.factories import result_record_factory


class TestAnalyzeOptionOrder:
    def test_no_results(self) -> None:
        analysis = analyze_option_order(results=[])

        assert analysis.options == ()
        assert analysis.cramer_v.shape == (0, 0)

    def test_one_result(self) -> None:
        result = result_record_factory()
        analysis = analyze_option_order([result])

        assert analysis.options == ((1,), (2,))
        assert analysis.cramer_v.shape == (2, 2)
        assert analysis.cramer_v[0, 1] == 0.0
        assert np.isnan(analysis.cramer_v).sum() == analysis.cramer_v.size - 1

    def test_no_order_effect(self) -> None:
        results = [result_record_factory() for _ in range(2)]
        results[1].comparison = ((2,), (1,))
        results[1].preferred_option_index = 1
        analysis = analyze_option_order(results=results)

        assert analysis.options == ((1,), (2,))
        assert analysis.cramer_v.shape == (2, 2)
        assert analysis.cramer_v[0, 1] == 0.0
        assert np.isnan(analysis.cramer_v).sum() == analysis.cramer_v.size - 1

    def test_perfect_order_effect(self) -> None:
        results = [result_record_factory() for _ in range(2)]
        results[1].comparison = ((2,), (1,))
        analysis = analyze_option_order(results=results)

        assert analysis.options == ((1,), (2,))
        assert analysis.cramer_v.shape == (2, 2)
        assert analysis.cramer_v[0, 1] == 1.0
        assert np.isnan(analysis.cramer_v).sum() == analysis.cramer_v.size - 1

    def test_partial_order_effect(self) -> None:
        results = [result_record_factory() for _ in range(3)]
        results[1].comparison = ((2,), (1,))
        results[2].comparison = ((2,), (1,))
        results[2].preferred_option_index = 1
        analysis = analyze_option_order(results=results)

        assert analysis.options == ((1,), (2,))
        assert analysis.cramer_v.shape == (2, 2)
        assert 0.0 < analysis.cramer_v[0, 1] < 1.0
        assert np.isnan(analysis.cramer_v).sum() == analysis.cramer_v.size - 1


class TestCompileObservations:
    def test_zero_results(self) -> None:
        observations = compile_observations([])
        matrix = observations.matrix

        assert observations.options == ()
        assert matrix.shape == (0, 0, NUM_OPTION_ORDERINGS, NUM_POSSIBLE_OUTCOMES)
        assert matrix.size == 0

    def test_one_result(self) -> None:
        result = result_record_factory()
        assert result.preferred_option_index is not None
        observations = compile_observations([result])
        matrix = observations.matrix

        assert observations.options == ((1,), (2,))
        assert matrix.shape == (2, 2, NUM_OPTION_ORDERINGS, NUM_POSSIBLE_OUTCOMES)
        assert matrix[0, 1, 0, result.preferred_option_index] == 1
        # All other elements should be 0
        assert matrix.sum() == 1

    def test_multiple_results(self) -> None:
        results = [result_record_factory() for _ in range(3)]
        results[1].comparison = ((2,), (1,))
        results[1].preferred_option_index = 1
        results[2].preferred_option_index = None
        observations = compile_observations(results)
        matrix = observations.matrix

        assert observations.options == ((1,), (2,))
        assert matrix.shape == (2, 2, NUM_OPTION_ORDERINGS, NUM_POSSIBLE_OUTCOMES)
        expected = np.array(
            [
                [1, 0, 1],
                [1, 0, 0],
            ],
        )
        assert np.all(matrix[0, 1, :, :] == expected)
        # All other elements should be 0
        assert matrix.sum() == expected.sum()
