import numpy as np
import pytest

from llmprefs.analysis.rating import RatedOptions, ValueCI, compile_matrix, rate_options
from llmprefs.task_structs import ResultRecord
from llmprefs.testing.factories import result_record_factory


@pytest.fixture
def mock_results() -> list[ResultRecord]:
    results = [result_record_factory() for _ in range(3)]
    results[0].comparison = ((0,), (1,))
    results[1].comparison = ((0,), (2,))
    results[2].comparison = ((1,), (2,))
    return results


def highest_rating(ratings: RatedOptions) -> ValueCI:
    return max(ratings.values(), key=lambda vci: vci.value)


class TestRateOptions:
    def test_zero_results(self) -> None:
        assert rate_options(results=(), num_resamples=1, confidence=0.0) == {}

    def test_one_result(self) -> None:
        result = result_record_factory()
        assert result.preferred_option_index is not None
        preferred_option = result.comparison[result.preferred_option_index]
        ratings = rate_options(
            results=[result],
            num_resamples=1,
            confidence=0.0,
        )

        assert len(ratings) == len(result.comparison)
        assert preferred_option in ratings
        assert highest_rating(ratings) == ratings[preferred_option]

    def test_multiple_results(self, mock_results: list[ResultRecord]) -> None:
        ratings = rate_options(mock_results, num_resamples=1, confidence=0.0)

        assert len(ratings) == len(mock_results)
        for i in range(1, len(mock_results)):
            assert ratings[(i - 1,)].value > ratings[(i,)].value


class TestCompileMatrix:
    def test_zero_results(self) -> None:
        option_matrix = compile_matrix(())
        assert len(option_matrix.options) == 0
        assert option_matrix.matrix.size == 0

    def test_one_result(self) -> None:
        result = result_record_factory()
        assert result.preferred_option_index is not None
        preferred_option = result.comparison[result.preferred_option_index]
        optmat = compile_matrix([result])

        assert len(optmat.options) == 2
        assert optmat.matrix.shape == (2, 2)
        assert optmat.matrix.sum() == 1
        for i, option in enumerate(optmat.options):
            assert optmat.matrix[i, i] == 0
            if option == preferred_option:
                assert optmat.matrix[i, 1 - i] == 1
            else:
                assert optmat.matrix[i, 1 - i] == 0

    def test_multiple_results(self, mock_results: list[ResultRecord]) -> None:
        optmat = compile_matrix(mock_results)

        assert optmat.options == ((0,), (1,), (2,))
        assert optmat.matrix.shape == (3, 3)
        expected_matrix = np.array(
            [
                [0, 1, 1],
                [0, 0, 1],
                [0, 0, 0],
            ]
        )
        assert (optmat.matrix == expected_matrix).all()
