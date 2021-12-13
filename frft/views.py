from django.shortcuts import render
from django.http import HttpResponse

import json
# Create your views here.

import numpy as np
import scipy
import scipy.signal as sg


def frft(f, a):
    """
    Calculate the fast fractional fourier transform.
    Parameters
    ----------
    f : numpy array
        The signal to be transformed.
    a : float
        fractional power
    Returns
    -------
    data : numpy array
        The transformed signal.
    References
    ---------
     .. [1] This algorithm implements `frft.m` from
        https://nalag.cs.kuleuven.be/research/software/FRFT/
    """
    ret = np.zeros_like(f, dtype=np.complex)
    f = f.copy().astype(np.complex)
    N = len(f)
    shft = np.fmod(np.arange(N) + np.fix(N / 2), N).astype(int)
    sN = np.sqrt(N)
    a = np.remainder(a, 4.0)

    # Special cases
    if a == 0.0:
        return f
    if a == 2.0:
        return np.flipud(f)
    if a == 1.0:
        ret[shft] = np.fft.fft(f[shft]) / sN
        return ret
    if a == 3.0:
        ret[shft] = np.fft.ifft(f[shft]) * sN
        return ret

    # reduce to interval 0.5 < a < 1.5
    if a > 2.0:
        a = a - 2.0
        f = np.flipud(f)
    if a > 1.5:
        a = a - 1
        f[shft] = np.fft.fft(f[shft]) / sN
    if a < 0.5:
        a = a + 1
        f[shft] = np.fft.ifft(f[shft]) * sN

    # the general case for 0.5 < a < 1.5
    alpha = a * np.pi / 2
    tana2 = np.tan(alpha / 2)
    sina = np.sin(alpha)
    f = np.hstack((np.zeros(N - 1), sincinterp(f), np.zeros(N - 1))).T

    # chirp premultiplication
    chrp = np.exp(-1j * np.pi / N * tana2 / 4 *
                     np.arange(-2 * N + 2, 2 * N - 1).T ** 2)
    f = chrp * f

    # chirp convolution
    c = np.pi / N / sina / 4
    ret = sg.fftconvolve(
        np.exp(1j * c * np.arange(-(4 * N - 4), 4 * N - 3).T ** 2),
        f
    )
    ret = ret[4 * N - 4:8 * N - 7] * np.sqrt(c / np.pi)

    # chirp post multiplication
    ret = chrp * ret

    # normalizing constant
    ret = np.exp(-1j * (1 - a) * np.pi / 4) * ret[N - 1:-N + 1:2]

    return ret


def ifrft(f, a):
    """
    Calculate the inverse fast fractional fourier transform.
    Parameters
    ----------
    f : numpy array
        The signal to be transformed.
    a : float
        fractional power
    Returns
    -------
    data : numpy array
        The transformed signal.
    """
    return frft(f, -a)


def sincinterp(x):
    N = len(x)
    y = np.zeros(2 * N - 1, dtype=x.dtype)
    y[:2 * N:2] = x
    xint = scipy.signal.fftconvolve(
        y[:2 * N],
        np.sinc(np.arange(-(2 * N - 3), (2 * N - 2)).T / 2),
    )
    return xint[2 * N - 3: -2 * N + 3]


def gen_signal(signal):
  input_signal = []

  if (signal == "box"):
    x = np.linspace(0, 120, 121)
    # Rect graph shape
    input_signal = np.zeros(121)
    for i in range(len(x)):
      if x[i] >= 50 and x[i]<=80:
        input_signal[i] = 1

  elif (signal == "triangle"):
    input_signal = sg.triang(121)

  elif (signal == "sin"):
    x = np.linspace(0,120,121)
    input_signal = np.sin(2 * np.pi * 1./60 * x)

  elif (signal == "cos"):
    # Create axis
    x = np.linspace(0,120,121)
    # Cos graph shape
    input_signal = np.cos(2 * np.pi * 1./60 * x)

  return input_signal
  
def calculate_frft(input_signal, fractional_power):
  real = np.real(frft(input_signal,fractional_power))
  imaginary = np.imag(frft(input_signal,fractional_power))

  return real, imaginary

def calculate_ft(input_signal):
  real = np.real(frft(input_signal,1))
  imaginary = np.imag(frft(input_signal,1))

  return real, imaginary


def calc_frft(request, signal_type, fractional_power):

    x = np.linspace(0,120,121)
    inp_sig = gen_signal(signal_type)

    real_frft_sig, img_frft_sig = calculate_frft(inp_sig,0.2)
    real_ft_sig, img_ft_sig = calculate_ft(inp_sig)

    response = {}

    response["real_frft"] = real_frft_sig.tolist()
    response["imag_frft"] = img_frft_sig.tolist()

    response["real_ft"] = img_ft_sig.tolist()
    response["imag_ft"] = img_ft_sig.tolist()

    return HttpResponse(json.dumps(response), content_type="application/json")