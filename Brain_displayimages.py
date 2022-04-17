from extractslices import extractslices
from extractpatches import extractpatches
import numpy as np
import scipy.spatial.distance as ssd 
from PIL import Image
from scipy.misc import toimage
import matplotlib.pyplot as plt
from PIL import ImageEnhance
import cv2
from hausdorff import hausdorff_distance


def displayimages(pred,orig,subject_no,slice_no,pred_csf,pred_gm,pred_wm,orig_csf,orig_gm,orig_wm,image_test_array):
    
    patch = subject_no*slice_no 
    Final_Pred = (pred[patch,:,:,1]*1)+(pred[patch,:,:,2]*2)+(pred[patch,:,:,3]*3)
    Final_Orig = (orig[patch,:,:,1]*1)+(orig[patch,:,:,2]*2)+(orig[patch,:,:,3]*3)
    
    Inter_GT = image_test_array[patch,:,:]
    d1,d2,d3 = Inter_GT.shape
    Final_GT = Inter_GT.reshape(d1,d2*d3)
    
    
    def combine_patches(mat):
        combine = np.flipud(mat[24:232,41:217])
        return combine
    
    Final_csf = (pred_csf[patch,:,:])
    Final_gm = (pred_gm[patch,:,:])
    Final_wm = (pred_wm[patch,:,:])
    Final_orig_csf = (orig_csf[patch,:,:])
    Final_orig_gm = (orig_gm[patch,:,:])
    Final_orig_wm = (orig_wm[patch,:,:])
    
    
    np.random.seed(0)
    print("Hausdorff distance CSF: {0}".format( hausdorff_distance( Final_csf, Final_orig_csf, distance="euclidean") ))
    print("Hausdorff distance GM: {0}".format( hausdorff_distance(Final_gm,Final_orig_gm, distance="euclidean") ))
    print("Hausdorff distance WM: {0}".format( hausdorff_distance(Final_wm, Final_orig_wm, distance="euclidean") ))
         
    
    plt.figure(figsize=(5, 5))
    plt.subplot(231)
    plt.imshow(combine_patches(Final_GT), cmap = 'gray', interpolation ='bicubic')    
    plt.title('Img')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.subplot(232)
    plt.imshow(combine_patches(Final_Orig), cmap = 'gray', interpolation ='bicubic')
    plt.title('Orig')
    plt.show()
    
    plt.figure(figsize=(5, 5))
    plt.subplot(233)
    plt.imshow(combine_patches(Final_Pred), cmap = 'gray', interpolation ='bicubic')
    plt.title('Pred')
    plt.show()
    
    plt.figure(figsize=(5, 5))
    plt.subplot(234)
    plt.imshow(combine_patches(Final_csf), cmap = 'gray', interpolation ='bicubic')    
    plt.title('CSF')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.subplot(235)
    plt.imshow(combine_patches(Final_gm), cmap = 'gray', interpolation ='bicubic')
    plt.title('GM')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.subplot(236)
    plt.imshow(combine_patches(Final_wm), cmap = 'gray', interpolation ='bicubic')
    plt.title('WM')
    plt.show()
