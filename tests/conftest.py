import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="Run external tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    run_external = config.getoption("--run-external")
    if run_external:
        return
    skip_external = pytest.mark.skip(reason="Missing --run-external flag")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip_external)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
