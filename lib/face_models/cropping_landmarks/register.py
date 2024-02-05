import numpy as np 
import scipy.io as sio
import cv2
import scipy.misc
import scipy.ndimage
import PIL.Image
import PIL.ImageFile
from PIL import Image

import numpy as np 
import scipy.io as sio
import cv2
import scipy

def getTransform( X , Y ):
    npts = X.shape[0]
    muX = np.mean(X,0)
    muY = np.mean(Y,0)
    X0 = X - np.ones((npts,1)) * muX
    Y0 = Y - np.ones((npts,1)) * muY
    normX = np.linalg.norm(X0)
    normY = np.linalg.norm(Y0)
    X0 = X0/ normX
    Y0 = Y0/ normY
    A = X0.transpose()*Y0
    L,D,M = np.linalg.svd(A)
    T = M * L.transpose()
    traceTA = np.sum(D)
    b = traceTA * normX/normY
    c = muX - b*muY*T
    return (T,b,c)

def kike_getT(pts, mShape, rotate=0):
    T = getTransform(mShape,pts)
    #print(T)
    T = list(T)
    scl = T[1]
    T[1] = 1/scl
    T[2] = 0 * T[2]
    if rotate == 0:
        T[0] = np.matrix([[1,0],[0,1]])
    ptsreg = kike_applyT( pts, T, 1)
    return T, ptsreg

def kike_applyT(pts, T, fwd=0):
    if fwd == 1:
        pts = (1/T[1]) * pts * T[0] + (1/T[1])* T[2] * (np.matrix([[1,0],[0,1]]) - T[0])
    else:
        pts = T[1] * pts * T[0].transpose() + T[2] * (np.matrix([[1,0],[0,1]]) - T[0].transpose())
    return pts

def registerImage_inverse( pts, mShape, rotation_angle, T, img, rot_mat, rotate = 0):
    npts = pts.shape[0]
    
    T,_ = kike_getT( np.matrix(pts), np.matrix(mShape), rotate)
    rotangle = rotation_angle #(180.0/np.pi) * np.arccos(T[0][0,0])
    # print(rotangle)
    # R = rot_mat[:, :2]
    # print(R)
    # T_inv = np.linalg.inv(R)
    if np.isnan(rotangle):
        rotangle = 0.0
    if np.sign( np.sin(rotangle * np.pi/180) ) != np.sign(T[0][1,0]):
        rotangle = -1.0 * rotangle
    if rotangle > 60:
        rotangle = 60
    if rotangle < -60:
        rotangle = -60
    if img is None:
        c = np.mean(pts,0)
        imreg = None
    else:
        c = np.array([img.shape[1]/2 + 0.5, img.shape[0]/2 + 0.5])#/T[1]
        rot_mat1 = cv2.getRotationMatrix2D(tuple(c), rotangle, 1.0)
        imreg = cv2.warpAffine(img, rot_mat, (int(img.shape[1]), int(img.shape[0])))
       
    T[2] = np.ones((66,1)) * np.matrix([[c[0],c[1]]])
    ptsreg = kike_applyT(pts,T,1) * T[1]
    return T, ptsreg, imreg

def registerImage( pts, mShape, img=None, angle = None, rotate=0):
    npts = pts.shape[0]
    T,_ = kike_getT( np.matrix(pts), np.matrix(mShape), rotate)
    if angle is not None:
        rotangle = angle
    else:
        rotangle = (180.0/np.pi) * np.arccos(T[0][0,0])
    if np.isnan(rotangle):
        rotangle = 0.0
    if np.sign( np.sin(rotangle * np.pi/180) ) != np.sign(T[0][1,0]):
        rotangle = -1.0 * rotangle
    if rotangle > 60:
        rotangle = 60
    if rotangle < -60:
        rotangle = -60
    if img is None:
        c = np.mean(pts,0)
        imreg = None
    else:
        c = np.array([img.shape[1]/2 + 0.5, img.shape[0]/2 + 0.5])#/T[1]
        rot_mat = cv2.getRotationMatrix2D(tuple(c), rotangle, 1.0)
        imreg = cv2.warpAffine(img, rot_mat, (int(img.shape[1]), int(img.shape[0])))

    T[2] = np.ones((66,1)) * np.matrix([[c[0],c[1]]])
    ptsreg = kike_applyT(pts,T,1) * T[1]
    return T, ptsreg, imreg, rotangle, T, rot_mat


def crop_scale(img, pts, res=256, scale_size = 200):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    # bb =  [0, 0, 255, 255]


    bb = [min_x, min_y, max_x, max_y]
    center = np.array([bb[2] - (bb[2]-bb[0])/2, bb[3] - (bb[3]-bb[1])/2])
    scale = (bb[2]-bb[0] + bb[3]-bb[1])/250.0
    # bb = [min_x, min_y, max_x, max_y]
    h = scale_size * scale
    # print(scale_size, scale, h)
    t = np.zeros((3, 3))
    t[0, 0] = float(res) / h
    t[1, 1] = float(res) / h
    t[0, 2] = res * (-float(center[0]) / h + .5)
    t[1, 2] = res * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    mat = t[:2]
    
    inp = cv2.warpAffine(img, mat, (res, res))
    pts = np.dot(np.concatenate((pts, pts[:, 0:1]*0+1), axis=1), mat.T)
    return inp, pts

