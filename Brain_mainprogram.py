from extractslices import extractslices
from extractpatches import extractpatches
from Unet import Unet
from keras.callbacks import EarlyStopping
from displayresults import displayresults
from displayimages import displayimages
import matplotlib.pyplot as plt

import numpy as np 
import scipy.spatial.distance as ssd 


(imgs_train_array, mask_train_array, image_test_array, mask_test_array) = extractslices()

windowsize_r = 256
windowsize_c = 256
num_patch = 1

model = Unet(windowsize_r,windowsize_c)

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=0.0001, restore_best_weights=True)
history = model.fit(imgs_train_array, mask_train_array, batch_size=1, epochs=10, verbose=1,
                  validation_split=0.2, shuffle=True,callbacks=[es])

masks_pred = model.predict(image_test_array, batch_size=1, verbose=1)
gm_dsc, gm_jc, wm_dsc, wm_jc, csf_dsc, csf_jc = displayresults(masks_pred,mask_test_array)

print("Gray Matter:",gm_dsc, gm_jc)
print("White Matter:",wm_dsc, wm_jc)
print("CSF:",csf_dsc, csf_jc)

subject_no = 1
slice_no = 23
displayimages(masks_pred,mask_test_array,subject_no,slice_no,pred_csf,pred_gm,pred_wm,orig_csf,orig_gm,orig_wm,image_test_array)
MSE = np.square(np.subtract(mask_test_array, masks_pred)).mean()
print("MSE:",MSE)

