from llmprefs.analysis.ranking import compile_matrix, rate_options


class TestRateOptions:
    def test_zero_results(self) -> None:
        assert rate_options(()) == {}


class TestCompileMatrix:
    def test_zero_results(self) -> None:
        option_matrix = compile_matrix(())
        assert len(option_matrix.options) == 0
        assert option_matrix.matrix.size == 0
