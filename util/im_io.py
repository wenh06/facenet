# -*- coding: utf-8 -*-
"""
updated image io function, to replace corr. deprecated functions in scipy.misc

NOT finished!
"""
try:
    import cv2
    imread = lambda im: cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
    imwrite = cv2.imwrite
    imrotate = cv2.rotate
    imresize = None
except ModuleNotFoundError:
    import imageio
    import skimage
    imread = imageio.imread
    imwrite = imageio.imwrite
    imrotate = skimage.transform.rotate
    imresize = None
