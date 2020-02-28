#from sklearn.datasets import  load_sample_image
from sklearn.feature_extraction import image
import cv2

one_image = cv2.imread('cancer.jpg')
print('Image shape: {}' . format(one_image.shape))

patches = image.extract_patches_2d(one_image, (227, 227), 100)
print('Patches shape: {}' .format(patches[0].shape))

for i in range(len(patches)):
    cv2.imshow('patch',patches[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()