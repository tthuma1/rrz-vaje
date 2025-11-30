import numpy as np
import matplotlib.pyplot as plt
import cv2

### f)
# Pogosto lahko obravnavamo problem iskanja krožnic, ko imamo radij že poznan.
# Kakšna je v tem primeru enačba, ki jo ena točka generira v parametričnem prostoru?
#     Enačba je enaka `r^2 = (x - x_c)^2 + (y - yc)^2`, kjer so `r`, `x` in `y` podani. V parametričnem prostoru
#     nam možne rešitve `x_c` in `y_c` torej opisujejo krožnico.
#
#     V parametričnem prostoru za posamezno točko krožnice dobimo vse točke, ki se od nje oddaljene za
#     določen radij. Torej za vsako točko na krožnici dobimo krožnico v parametričnem prostoru. Pravo središče
#     krožnice je točka v parametričnem prostoru, skozi katero gre največ krožnic v parametričnem prostoru
#     (točka, ki prejme največ glasov).