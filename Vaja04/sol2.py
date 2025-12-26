import cv2
import matplotlib.pyplot as plt
import numpy as np
from workspace_utils import *

### a)

ime_slike = 'capture_f1.jpg'
im = cv2.imread(ime_slike)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

corners = get_workspace_corners(im)
print(corners)


