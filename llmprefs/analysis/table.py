from dataclasses import dataclass

from tabulate import tabulate

from llmprefs.analysis.rating import ComparisonOutcomes, RatedOptions


@dataclass
class TableRow:
    option: str
    wins: int
    losses: int
    draws: int
    rating_lower: float
    rating_mid: float
    rating_upper: float


def print_table(
    outcomes: ComparisonOutcomes,
    rated_options: RatedOptions,
    table_format: str,
    confidence: float,
) -> None:
    table = assemble_table(outcomes, rated_options, confidence)
    print(  # noqa: T201
        tabulate(
            # `table` is a list of dataclasses, which the tabulate documentation
            # explicitly allows. But the type hints apparently don't reflect this.
            table,  # pyright: ignore [reportArgumentType]
            headers=[
                "Option ID",
                "Wins",
                "Losses",
                "Draws",
                "Rating CI Lower",
                "Rating CI Mid",
                "Rating CI Upper",
            ],
            tablefmt=table_format,
            floatfmt=".3f",
        )
    )


def assemble_table(
    outcomes: ComparisonOutcomes,
    rated_options: RatedOptions,
    confidence: float,
) -> list[TableRow]:
    if outcomes.options != rated_options.options:
        raise ValueError("Outcome and rating options must be identical")
    options = outcomes.options
    rating_values = rated_options.values(confidence)
    rows: list[TableRow] = []
    for i, opt in enumerate(options):
        rows.append(
            TableRow(
                option=str(opt[0]) if len(opt) == 1 else str(opt),
                wins=outcomes.counts[i, :, 0].sum() + outcomes.counts[:, i, 1].sum(),
                losses=outcomes.counts[i, :, 1].sum() + outcomes.counts[:, i, 0].sum(),
                draws=outcomes.counts[i, :, 2].sum() + outcomes.counts[:, i, 2].sum(),
                rating_lower=rating_values[opt].ci_lower,
                rating_mid=rating_values[opt].value,
                rating_upper=rating_values[opt].ci_upper,
            )
        )
    return rows
