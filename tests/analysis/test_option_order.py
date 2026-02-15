import numpy as np

from llmprefs.analysis.option_order import analyze_option_order
from llmprefs.testing.factories import result_record_factory


class TestAnalyzeOptionOrder:
    def test_no_results(self) -> None:
        analysis = analyze_option_order(results=[])

        assert analysis.options == ()
        assert analysis.deltas.shape == (0, 0)

    def test_one_result(self) -> None:
        result = result_record_factory()
        analysis = analyze_option_order([result])

        assert analysis.options == ((1,), (2,))
        assert analysis.deltas.shape == (2, 2)
        assert analysis.deltas[0, 1] == 1.0
        assert np.isnan(analysis.deltas).sum() == analysis.deltas.size - 1

    def test_no_order_effect(self) -> None:
        results = [result_record_factory() for _ in range(2)]
        results[1].comparison = ((2,), (1,))
        results[1].preferred_option_index = 1
        analysis = analyze_option_order(results=results)

        assert analysis.options == ((1,), (2,))
        assert analysis.deltas.shape == (2, 2)
        assert analysis.deltas[0, 1] == 0.0
        assert np.isnan(analysis.deltas).sum() == analysis.deltas.size - 1

    def test_perfect_order_effect(self) -> None:
        results = [result_record_factory() for _ in range(2)]
        results[1].comparison = ((2,), (1,))
        analysis = analyze_option_order(results=results)

        assert analysis.options == ((1,), (2,))
        assert analysis.deltas.shape == (2, 2)
        assert analysis.deltas[0, 1] == 1.0
        assert np.isnan(analysis.deltas).sum() == analysis.deltas.size - 1

    def test_partial_order_effect(self) -> None:
        results = [result_record_factory() for _ in range(3)]
        results[1].comparison = ((2,), (1,))
        results[2].comparison = ((2,), (1,))
        results[2].preferred_option_index = 1
        analysis = analyze_option_order(results=results)

        # Our mock results look like this, where * indicates the preferred option:
        # *1  2
        # *2  1
        #  2 *1
        # So there's a slight preference for the option which is presented first.
        assert analysis.options == ((1,), (2,))
        assert analysis.deltas.shape == (2, 2)
        assert np.isclose(analysis.deltas[0, 1], 1.0 / 3.0)
        assert np.isnan(analysis.deltas).sum() == analysis.deltas.size - 1
