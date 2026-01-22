from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
path="/Users/manivannans/Dino_Pathology/data/NCT-CRC-HE-100K/ADI/ADI-AAAMHQMK.tif"
image=Image.open(path)
plt.imshow(np.array(image))
plt.axis("off")
plt.show()