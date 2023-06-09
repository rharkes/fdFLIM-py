from typing import Any, Tuple
import numpy as np


def phimoddc(
    stack: np.ndarray[Any, Any], axis: int = 2
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
    mod = (2 * (np.sqrt(fs**2 + fc**2))) / dc
    phi = np.arctan2(fc, fs)
    return phi, mod, dc


def getomega(f: float) -> float:
    """
    convert a frequency to an angular frequency
    :param f: frequency(Hz)
    :return: angular frequency(Hz)
    """
    return f * 2 * np.pi


def phi2tau(phi: np.ndarray[Any, Any], f: float) -> np.ndarray[Any, Any]:
    """
    get the lifetime from the phase shift
    :param phi: phase shift
    :param f: frequency (Hz)
    :return: lifetime (s)
    """
    tau_phi = (1 / getomega(f)) * np.tan(phi)
    return np.array(tau_phi)


def mod2tau(mod: np.ndarray[Any, Any], f: float) -> np.ndarray[Any, Any]:
    """
    get the lifetime from the de-modulation
    :param mod: de-modulation
    :param f: frequency (Hz)
    :return: lifetime(s)
    """
    tau_mod = (1 / getomega(f)) * np.sqrt((1 / (mod**2)) - 1)
    return np.array(tau_mod)


def tau2mod(tau: float, f: float) -> float:
    """
    get de-modulation from lifetime
    :param tau: lifetime(s)
    :param f: frequency(Hz)
    :return: de-modulation
    """
    mod = np.sqrt(1 / ((getomega(f) * tau) ** 2 + 1))
    return float(mod)


def tau2phi(tau: float, f: float) -> float:
    """
    get phase-shift from lifetime
    :param tau: lifetime(ns)
    :param f: frequency(Hz)
    :return: phase-shift
    """
    phi = np.arctan(getomega(f) * tau)
    return float(phi)


def getsystemphimod(
    stack: np.ndarray[Any, Any], tau: float, f: float, axis: int = 2
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
    phi, mod, dc = phimoddc(stack, axis=axis)  # measured phi and mod
    systemphi = ref_phi - phi
    systemmod = mod / ref_mod
    return systemphi, systemmod
