import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from numpy.typing import NDArray


def annotated_heatmap(
    axes: Axes,
    matrix: NDArray[np.float64],
    precision: int = 3,
) -> AxesImage:
    expected_dimensionality = 2
    if matrix.ndim != expected_dimensionality:
        raise ValueError("Matrix has wrong number of dimensions")

    image = axes.imshow(  # pyright: ignore[reportUnknownMemberType]
        matrix,
        cmap="viridis",
    )

    threshold = image.norm(np.nanmax(matrix)) / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            color = "white" if image.norm(value) < threshold else "black"
            axes.text(  # pyright: ignore[reportUnknownMemberType]
                x=j,
                y=i,
                s=f"{value:.{precision}f}",
                ha="center",
                va="center",
                color=color,
                size=6,
            )
    return image
