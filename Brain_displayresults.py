import numpy as np
from sklearn.metrics import confusion_matrix
from metrics import metrics

def displayresults(masks_pred,test_mask):
    masks_pred[masks_pred > 0.5] = 1
    masks_pred[masks_pred < 1.0] = 0
    pred_csf = np.round(masks_pred[:,:,:,1]).astype('uint16')
    pred_gm = np.round(masks_pred[:,:,:,2]).astype('uint16')
    pred_wm = np.round(masks_pred[:,:,:,3]).astype('uint16')
    
    orig_csf = test_mask[:,:,:,1]
    orig_gm = test_mask[:,:,:,2]
    orig_wm = test_mask[:,:,:,3]
    
    cnf_matrix_gm = confusion_matrix(orig_gm.flatten(), pred_gm.flatten())
    cnf_matrix_wm = confusion_matrix(orig_wm.flatten(), pred_wm.flatten())
    cnf_matrix_csf = confusion_matrix(orig_csf.flatten(), pred_csf.flatten())
    
    gm_dsc, gm_jc = metrics(cnf_matrix_gm)
    wm_dsc, wm_jc = metrics(cnf_matrix_wm)
    csf_dsc, csf_jc = metrics(cnf_matrix_csf)
    
    return (gm_dsc, gm_jc, wm_dsc, wm_jc,csf_dsc, csf_jc,pred_csf,pred_gm,pred_wm,orig_csf,orig_gm,orig_wm)