import pickle
from register import *
import matplotlib.pyplot as plt

data = pickle.load(open('data_register.pkl','rb'))
pts = data['pts']
img = data['img']
mShape = data['mShape']

Tout, ptsout, imout = registerImage( pts, mShape, img, rotate=0)
img_crop, pts_crop = crop(imout,ptsout.getA())
plt.imshow(img_crop)
plt.plot(pts_crop[:,0],pts_crop[:,1],'x')

plt.show()

Tout, ptsout, imout = registerImage( pts, mShape, img, rotate=1)
img_crop, pts_crop = crop(imout,ptsout.getA())
plt.imshow(img_crop)
plt.plot(pts_crop[:,0],pts_crop[:,1],'x')

plt.show()