import os
import sys
import time
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import torch

print(sys.path[0])
sys.path.append(os.path.join(sys.path[0],'tacotron2','waveglow'))
sys.path.append(os.path.join(sys.path[0],'tacotron2'))

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from denoiser import Denoiser
