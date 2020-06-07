"""
Several functions that are used for frequency domain lifetime imaging.
"""
import numpy as np


def phimoddc(stack, axis=2):
    """
    get phase shift, modulation and dc from a datastack
    :param stack: ndarray
    :param axis: phase axis
    :return: phi,mod,dc
    """
    s = stack.shape
    n_phase = s[axis]
    phase_vector = 2 * np.pi * (np.arange(n_phase) / n_phase)
    stack = stack.swapaxes(len(s) - 1, axis)  # put phase at the end
    axis = len(s) - 1
    fs = np.mean(stack * np.sin(phase_vector), axis=axis)
    fc = np.mean(stack * np.cos(phase_vector), axis=axis)
    dc = stack.mean(axis=axis)
    mod = (2 * (np.sqrt(fs ** 2 + fc ** 2))) / dc
    phi = np.arctan2(fc, fs)
    return phi, mod, dc


def getomega(f):
    """
    convert a frequency to an angular frequency
    :param f: frequency(Hz)
    :return: angular frequency(Hz)
    """
    return f * 2 * np.pi


def phi2tau(phi, f):
    """
    get the lifetime from the phase shift
    :param phi: phase shift
    :param f: frequency (Hz)
    :return: lifetime (s)
    """
    tau_phi = (1 / getomega(f)) * np.tan(phi)
    return tau_phi


def mod2tau(mod, f):
    """
    get the lifetime from the de-modulation
    :param mod: de-modulation
    :param f: frequency (Hz)
    :return: lifetime(s)
    """
    tau_mod = (1 / getomega(f)) * np.sqrt((1 / (mod ** 2)) - 1)
    return tau_mod


def tau2mod(tau, f):
    """
    get de-modulation from lifetime
    :param tau: lifetime(s)
    :param f: frequency(Hz)
    :return: de-modulation
    """
    mod = 1 / np.sqrt(1 / ((getomega(f) * tau) ** 2 + 1))
    return mod


def tau2phi(tau, f):
    """
    get phase-shift from lifetime
    :param tau: lifetime(ns)
    :param f: frequency(Hz)
    :return: phase-shift
    """
    phi = np.atan(getomega(f) * tau)
    return phi


def getsystemphimod(stack, tau, f, axis=2):
    """
    Get system phi and mod from a reference stack and reference lifetime.
    :param stack: reference stack
    :param tau: reference lifetime (ns)
    :param f: frequency (Hz)
    :param axis: phase axis
    :return:
    """
    ref_mod = tau2mod(tau, f)  # expected modulation
    ref_phi = tau2phi(tau, f)  # expected phase shift
    phi, mod = phimoddc(stack, f, axis=axis)  # measured phi and mod
    systemphi = phi - ref_phi
    systemmod = mod / ref_mod
    return systemphi, systemmod


class Reference:
    """
    Holds the systemphi and systemmod for a reference stack
    """

    def __init__(self, stack, tau, f, axis=2):
        self.f = f
        self.systemphi, self.systemmod = getsystemphimod(stack, tau, f, axis=axis)


class Sample:
    """
    Holds a sample and contains different ways to display it.
    Needs a reference measurement to calculate the real phi and mod.
    To just calculate phi and mod of a stack, use phimoddc.
    """

    def __init__(self, stack, reference, f=None, axis=2):
        if f is None:
            self.f = reference.f
        else:
            self.f = f
        if not self.f == reference.f:
            print('frequency of sample not equal to frequency of reference')
        phi, mod = phimoddc(stack, f, axis=axis)
        self.phi = phi + reference.systemphi
        self.mod = mod / reference.systemphi

    def getlifetimephase(self):
        """
        Get sample lifetime from phase
        :return: lifetime(s)
        """
        return phi2tau(self.phi)

    def getlifetimemod(self):
        """
        Get sample lifetime from modulation
        :return: lifetime(s)
        """
        return mod2tau(self.mod)

    def getphasorplotcoords(self):
        """
        Get all coordinates in a phaserplot
        :return: nparray with coordinates
        """
        coords = np.empty((self.mod.size, 2))
        coords[:, 0] = self.mod.flatten() * np.cos(self.phi.flatten())
        coords[:, 1] = self.mod.flatten() * np.sin(self.phi.flatten())
        return coords
