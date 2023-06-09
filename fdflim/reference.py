from typing import Any
import numpy as np
from .util import getsystemphimod


class Reference:
    """
    Holds the systemphi and systemmod for a reference stack
    """

    def __init__(
        self, stack: np.ndarray[Any, Any], tau: float, f: float, axis: int = 2
    ):
        self.f = f  # type:float
        self.systemphi, self.systemmod = getsystemphimod(stack, tau, f, axis=axis)
