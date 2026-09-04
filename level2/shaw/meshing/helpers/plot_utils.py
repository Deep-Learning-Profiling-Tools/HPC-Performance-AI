#!/usr/bin/env python

# HPC-Performance-AI: matplotlib is only needed for the optional plotting
# modes (-plotting != none); make it optional so meshes can be generated
# in environments without matplotlib.
try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None
import sys, os, time
import numpy as np
from numpy import linspace, meshgrid
try:
  from matplotlib import cm
except ImportError:
  cm = None

from earth_params import earthRadius, mantleThickness, cmbRadius

#----------------------------------------------------------------------------------------
def plotCMB(ax):
  # trace the CMB
  cmbTh = np.linspace(0, 2*np.pi, 100)
  cmbRa = cmbRadius*np.ones(100)
  ax.plot(cmbTh, cmbRa, c='b', linestyle='--', linewidth=0.5)

def plotEarthSurf(ax):
  # trace the earth surface
  surfTh = np.linspace(0, 2*np.pi, 100)
  surfRa = earthRadius*np.ones(100)
  ax.plot(surfTh, surfRa, c='b', linewidth=0.5)
