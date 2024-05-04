# Datasets



## CelebA-HQ

### Brief description

[**CelebFaces Attributes HQ Dataset (CelebA-HQ)**](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a large-scale face attributes dataset that consists of 30,000 high-resolution face images selected from the CelebA dataset, each of which annotated according to the following **40** attributes:

```
no_beard, young, attractive, mouth_slightly_open, smiling, wearing_lipstick, high_cheekbones, male, heavy_makeup, wavy_hair, oval_face, pointy_nose, arched_eyebrows, big_lips, black_hair, big_Nose, brown_hair, straight_hair, bags_under_eyes, wearing_earrings, bangs, blond_hair, bushy_eyebrows, narrow_eyes, wearing_necklace, 5_o_clock_shadow, receding_hairline, wearing_necktie, rosy_cheeks, eyeglasses, goatee, chubby, sideburns, blurry, wearing_hat, double_chin, pale_skin, gray_hair, mustache, bald
```

### Download

Download the dataset following the instructions given [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (e.g., following this [link](https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv) in Google Drive). The root directory of the dataset (`dataset_root`) should have the following structure:

```
├── annotations
│   ├── CelebA-HQ-to-CelebA-mapping.txt
│   ├── CelebAMask-HQ-attribute-anno.txt
│   └── list_eval_partition.txt
└── data
    ├── 0.jpg
    ├── 1.jpg
    ├── ...
    ├── 9998.jpg
    └── 9999.jpg
```



## LFW

### Brief description

[ **Labeled Faces in the Wild (LFW)**](http://vis-www.cs.umass.edu/lfw/) is a database of face photographs designed for studying the problem of unconstrained face recognition. The data set contains **more than 13,000 images of faces collected from the web**. Each face has been **labeled with the name of the person pictured**. **1680 of the people pictured have two or more distinct photos in the data set.** The only constraint on these faces is that they were detected by the Viola-Jones face detector. 

There are now four different sets of LFW images including the original and three different types of "aligned" images. The aligned images include "funneled images" (ICCV 2007), LFW-a, which uses an unpublished method of alignment, and "deep funneled" images (NIPS 2012). Among these, LFW-a and the deep funneled images produce superior results  for most face verification  algorithms over the original images and over the funneled images (ICCV 2007). 

### Download

Download the dataset following the instructions given [here](http://vis-www.cs.umass.edu/lfw/). The root directory of the dataset (`dataset_root`) should have the following structure:

```
├── annotations
│   └── lfw-names.txt
└── data
   ├── AJ_Cook
   │   └── AJ_Cook_0001.jpg
   ├── AJ_Lamas
   │   └── AJ_Lamas_0001.jpg
   ├── Aaron_Eckhart
   │   └── Aaron_Eckhart_0001.jpg
   ├── Aaron_Guiel
   │   └── Aaron_Guiel_0001.jpg
   ├── Aaron_Patterson
   │   └── Aaron_Patterson_0001.jpg
   ├── Aaron_Peirsol
   │   ├── Aaron_Peirsol_0001.jpg
   │   ├── Aaron_Peirsol_0002.jpg
   │   ├── Aaron_Peirsol_0003.jpg
   │   └── Aaron_Peirsol_0004.jpg
   ├── Aaron_Pena
   │   └── Aaron_Pena_0001.jpg
   ├── ...
   ├── Zumrati_Juma
   │   └── Zumrati_Juma_0001.jpg
   ├── Zurab_Tsereteli
   │   └── Zurab_Tsereteli_0001.jpg
   ├── Zydrunas_Ilgauskas
   │   └── Zydrunas_Ilgauskas_0001.jpg
   └── lfw-names.txt
```



**Notes:**

http://vis-www.cs.umass.edu/lfw/lfw.tgz

http://vis-www.cs.umass.edu/lfw/lfw-names.txt

`tar -xzvf lfw.tgz`

