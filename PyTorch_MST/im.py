from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

dir1 = r"/homes/h19savat/Documents/Pytorch_MST/style/noisy/candy_noise100.jpg"
dir2 = r"/homes/h19savat/Documents/Pytorch_MST/style/candy.jpg"

im1 = np.array(Image.open(dir1))
im2 = np.array(Image.open(dir2))

im1 = np.dot(im1[...,:3], [0.2989, 0.5870, 0.1140])
im2 = np.dot(im2[...,:3], [0.2989, 0.5870, 0.1140])

#dif = im1 - im2

if __name__ == "__main__":

    print(ssim(im1, im2))
    #plt.imshow(np.abs(dif), cmap="gray")
    #plt.show()