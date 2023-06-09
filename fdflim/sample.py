import math
from typing import Any
import numpy as np
from .reference import Reference
from .util import phimoddc, phi2tau, mod2tau


class Sample:
    """
    Holds a sample and contains different ways to display it.
    Needs a reference measurement to calculate the real phi and mod.
    To just calculate phi and mod of a stack, use phimoddc.
    """

    def __init__(
        self,
        stack: np.ndarray[Any, Any],
        reference: Reference,
        f: float = float("nan"),
        axis: int = 2,
    ) -> None:
        if math.isnan(f):
            self.f = reference.f
        else:
            self.f = f
        if not self.f == reference.f:
            print("frequency of sample not equal to frequency of reference")
        self.s = stack.shape
        phi, mod, dc = phimoddc(stack, axis=axis)
        systemphi = reference.systemphi
        systemmod = reference.systemmod
        for i in range(2, len(phi.shape)):
            systemphi = systemphi[:, :, np.newaxis]
            systemmod = systemmod[:, :, np.newaxis]
        self.phi = phi + systemphi
        self.mod = mod / systemmod
        self.dc = dc

    def getlifetimephase(self) -> np.ndarray[Any, Any]:
        """
        Get sample lifetime from phase
        :return: lifetime(s)
        """
        return phi2tau(self.phi, self.f)

    def getlifetimemod(self) -> np.ndarray[Any, Any]:
        """
        Get sample lifetime from modulation
        :return: lifetime(s)
        """
        return mod2tau(self.mod, self.f)

    def getphasorplotcoords(self) -> np.ndarray[Any, Any]:
        """
        Get all coordinates in a phaserplot
        :return: nparray with coordinates
        """
        coords = np.empty((self.mod.size, 2))
        coords[:, 0] = self.mod.flatten() * np.cos(self.phi.flatten())
        coords[:, 1] = self.mod.flatten() * np.sin(self.phi.flatten())
        return coords
