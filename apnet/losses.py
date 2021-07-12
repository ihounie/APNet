from keras import backend as K
from tensorflow.nn import top_k


def prototype_loss(y_true, y_pred):
    print('loss',y_pred.shape)
    if len(y_pred.shape) == 3:
        y_pred = K.mean(y_pred,axis=-1)
    if len(y_pred.shape) == 4:
        y_pred = K.mean(y_pred,axis=(2,3))
    print('loss',y_pred.shape)

    error_1 = K.mean(K.min(y_pred, axis = 0))
    error_2 = K.mean(K.min(y_pred, axis = 1))
    return 1.0*error_1 + 1.0*error_2

def separate_loss(y_true, y_pred):
    L2_pair = (K.expand_dims(y_pred, 0)-K.expand_dims(y_pred, 1))**2
    print('L2 pair', L2_pair.shape)
    s_loss =  - K.mean(L2_pair)
    #print('sep input',y_pred.shape)
    #s_loss = - K.mean(y_pred[None, :, :], axis=(1,2))
    return s_loss

def separate_loss_inv(y_true, y_pred):
    print('sep loss',y_pred.shape)
    #Calculate distance inverse
    # Add an eps to avoid overflow
    distance_inverse = 1/(y_pred+1e-4)
    # Loss is mean inverse disatance
    loss = K.mean(distance_inverse, axis=(0,1))[None,:]
    print('sep loss',loss.shape)
    return loss
