from dataclasses import dataclass
from typing import Sequence
import numpy as np
import cv2


@dataclass
class BirdsImage:
    """Input image with birds and their contours."""
    filename: str
    image: np.ndarray | None = None
    contours: Sequence[np.ndarray] | None = None
    thresh: int = 127
    _is_fit: bool = False

    @property
    def image_orig(self) -> np.ndarray:
        """Original, not modified image."""
        return cv2.imread(self.filename)

    @property
    def count(self) -> int:
        """Number of found birds in the image."""
        if not self._is_fit:
            raise ValueError('run `.fit()` first')
        return len(self.contours)

    def fit(self) -> None:
        """Identify birds in the image."""
        gray = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY)
        self.image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, 1)
        self.contours, _ = cv2.findContours(threshold, 1, 2)
        self._is_fit = True

    def annotate(self) -> np.ndarray:
        """Draw contours around identified birds."""
        if not self._is_fit:
            raise ValueError('run `.fit()` first')
            
        image = self.image.copy()

        for cnt in self.contours:
            cv2.drawContours(image, [cnt], 0, (0, 0, 255), 1)

        return image

    def side_by_side(self) -> np.ndarray:
        """Side by side comparison of the fitted image and the original one."""
        annotated = self.annotate()
        return np.hstack([annotated, self.image_orig])
            