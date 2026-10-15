import numpy as np


class RandomFlipRotate:
    """Apply identical random transforms to SAR and label."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
            self,
            image: np.ndarray,
            mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        # Horizontal flip
        if np.random.random() < self.p:
            image = np.flip(image, axis=2)
            mask = np.flip(mask, axis=1)

        # Vertical flip
        if np.random.random() < self.p:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=0)

        # Random 90-degree rotation
        k = np.random.randint(0, 4)

        image = np.rot90(image, k, axes=(1, 2))
        mask = np.rot90(mask, k, axes=(0, 1))

        return (
            np.ascontiguousarray(image),
            np.ascontiguousarray(mask),
        )
