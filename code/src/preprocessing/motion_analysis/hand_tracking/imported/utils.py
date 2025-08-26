import cv2
import numpy as np
import os
import pickle

# --- MONKEY-PATCH START ---
# Modification of the initial code by Basil Duvernoy
# Date: 2025-08-22
# The chumpy library expects these deprecated aliases to exist in NumPy.
# We are manually adding them back to the numpy namespace before chumpy is imported.
# This is a forward-compatible "patch" that avoids downgrading NumPy.
# Note: This should be done BEFORE the library causing the error is imported.
# In this case, pickle.load() will implicitly import chumpy.
np.int = int
np.float = float
np.bool = bool
np.complex = complex
np.object = object
np.unicode = str
np.str = str
# --- MONKEY-PATCH END ---

def imresize(img, size):
  """
  Resize an image with cv2.INTER_LINEAR.

  Parameters
  ----------
  size: (width, height)

  """
  return cv2.resize(img, size, cv2.INTER_LINEAR)


def load_pkl(path):
  """
  Load pickle data.

  Parameter
  ---------
  path: Path to pickle file.

  Return
  ------
  Data in pickle file.

  """
  if not os.path.exists(path):
    path =  os.path.abspath(os.path.join(__file__, '..', path))
  with open(path, 'rb') as f:
    data = pickle.load(f)
    # f = open(path, 'rb')
  return data


class LowPassFilter:
  def __init__(self):
    self.prev_raw_value = None
    self.prev_filtered_value = None

  def process(self, value, alpha):
    if self.prev_raw_value is None:
      s = value
    else:
      s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
    self.prev_raw_value = value
    self.prev_filtered_value = s
    return s


class OneEuroFilter:
  def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
    self.freq = freq
    self.mincutoff = mincutoff
    self.beta = beta
    self.dcutoff = dcutoff
    self.x_filter = LowPassFilter()
    self.dx_filter = LowPassFilter()

  def compute_alpha(self, cutoff):
    te = 1.0 / self.freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

  def process(self, x):
    prev_x = self.x_filter.prev_raw_value
    dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
    edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
    cutoff = self.mincutoff + self.beta * np.abs(edx)
    return self.x_filter.process(x, self.compute_alpha(cutoff))
