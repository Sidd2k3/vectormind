"""PCA: project high-dimensional vectors to 2D for the scatter plot.

Used for visualization only. Search always runs on the full vectors.
"""

import numpy as np


class PCA2D:
    def __init__(self, vectors: list[np.ndarray]):
        self.mean = None
        self.components = None
        if len(vectors) < 2:
            return
        X = np.vstack(vectors).astype(np.float64)
        self.mean = X.mean(axis=0)
        # SVD of the centred data: rows of Vt are the principal directions,
        # ordered by how much of the data's spread they capture.
        _, _, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        components = Vt[:2]
        # SVD signs are arbitrary; fix them so the plot doesn't flip around.
        for row in components:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        if components.shape[0] < 2:
            components = np.vstack([components, np.zeros_like(components[0])])
        self.components = components

    def project(self, v: np.ndarray) -> tuple[float, float]:
        if self.components is None:
            return (0.0, 0.0)
        x, y = self.components @ (np.asarray(v, dtype=np.float64) - self.mean)
        return (float(x), float(y))
