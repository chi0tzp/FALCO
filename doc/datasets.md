# Datasets



## CelebA

### Brief description

[**CelebFaces Attributes Dataset (CelebA)**](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a large-scale face attributes dataset with more than **200K** celebrity images, each with **40** attribute annotations. The images in this dataset cover large pose  variations and background clutter. CelebA has large diversities, large  quantities, and rich annotations, including

- **10,177** number of **identities**,
- **202,599** number of **face images**, and
- **5 landmark locations**, **40 binary attributes** annotations per image.

The dataset can be employed as the training and test sets for the  following computer vision tasks: face attribute recognition, face  recognition, face detection, landmark (or facial part) localization, and face editing & synthesis.      

Attributes frequencies on various  dataset subsets:

| **Attribute**         | **Train** | **Val** | **Test** | **Train+Val** | **Train+Val+Test** |
| --------------------- | --------- | ------- | -------- | ------------- | ------------------ |
| *no_beard*            | 0.83      | 0.82    | 0.85     | 0.83          | 0.83               |
| *young*               | 0.78      | 0.75    | 0.76     | 0.78          | 0.77               |
| *attractive*          | 0.51      | 0.52    | 0.50     | 0.51          | 0.51               |
| *mouth_slightly_open* | 0.48      | 0.48    | 0.50     | 0.48          | 0.48               |
| *smiling*             | 0.48      | 0.48    | 0.50     | 0.48          | 0.48               |
| *wearing_lipstick*    | 0.47      | 0.45    | 0.52     | 0.47          | 0.47               |
| *high_cheekbones*     | 0.45      | 0.45    | 0.48     | 0.45          | 0.46               |
| *male*                | 0.42      | 0.43    | 0.39     | 0.42          | 0.42               |
| *heavy_makeup*        | 0.38      | 0.39    | 0.40     | 0.38          | 0.39               |
| *wavy_hair*           | 0.32      | 0.28    | 0.36     | 0.31          | 0.32               |
| *oval_face*           | 0.28      | 0.28    | 0.30     | 0.28          | 0.28               |
| *pointy_nose*         | 0.28      | 0.28    | 0.29     | 0.28          | 0.28               |
| *arched_eyebrows*     | 0.27      | 0.26    | 0.28     | 0.27          | 0.27               |
| *big_lips*            | 0.24      | 0.15    | 0.33     | 0.23          | 0.24               |
| *black_hair*          | 0.24      | 0.21    | 0.27     | 0.24          | 0.24               |
| *big_Nose*            | 0.24      | 0.25    | 0.21     | 0.24          | 0.23               |
| *brown_hair*          | 0.20      | 0.24    | 0.18     | 0.21          | 0.21               |
| *straight_hair*       | 0.21      | 0.21    | 0.21     | 0.21          | 0.21               |
| *bags_under_eyes*     | 0.20      | 0.21    | 0.20     | 0.20          | 0.20               |
| *wearing_earrings*    | 0.19      | 0.19    | 0.21     | 0.19          | 0.19               |
| *bangs*               | 0.15      | 0.15    | 0.16     | 0.15          | 0.15               |
| *blond_hair*          | 0.15      | 0.15    | 0.13     | 0.15          | 0.15               |
| *bushy_eyebrows*      | 0.14      | 0.14    | 0.13     | 0.14          | 0.14               |
| *narrow_eyes*         | 0.12      | 0.08    | 0.15     | 0.11          | 0.12               |
| *wearing_necklace*    | 0.12      | 0.12    | 0.14     | 0.12          | 0.12               |
| *5_o_clock_shadow*    | 0.11      | 0.12    | 0.10     | 0.11          | 0.11               |
| *receding_hairline*   | 0.08      | 0.07    | 0.09     | 0.08          | 0.08               |
| *wearing_necktie*     | 0.07      | 0.07    | 0.07     | 0.07          | 0.07               |
| *rosy_cheeks*         | 0.07      | 0.07    | 0.07     | 0.07          | 0.07               |
| *eyeglasses*          | 0.07      | 0.07    | 0.07     | 0.07          | 0.07               |
| *goatee*              | 0.06      | 0.07    | 0.05     | 0.07          | 0.06               |
| *chubby*              | 0.06      | 0.06    | 0.05     | 0.06          | 0.06               |
| *sideburns*           | 0.06      | 0.07    | 0.05     | 0.06          | 0.06               |
| *blurry*              | 0.05      | 0.05    | 0.05     | 0.05          | 0.05               |
| *wearing_hat*         | 0.05      | 0.05    | 0.04     | 0.05          | 0.05               |
| *double_chin*         | 0.05      | 0.05    | 0.05     | 0.05          | 0.05               |
| *pale_skin*           | 0.04      | 0.04    | 0.04     | 0.04          | 0.04               |
| *gray_hair*           | 0.04      | 0.05    | 0.03     | 0.04          | 0.04               |
| *mustache*            | 0.04      | 0.05    | 0.04     | 0.04          | 0.04               |
| *bald*                | 0.02      | 0.02    | 0.02     | 0.02          | 0.02               |

### Download

Download the dataset following the instructions given [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The root directory of the dataset (`dataset_root`) should have the following structure:

```
├── Anno
│   ├── identity_CelebA.txt
│   ├── list_attr_celeba.txt
│   ├── list_bbox_celeba.txt
│   ├── list_landmarks_align_celeba.txt
│   └── list_landmarks_celeba.txt
├── Eval
│   └── list_eval_partition.txt
└── Img
    └── img_align_celeba
        ├── 000001.jpg
        ├── 000002.jpg
        ├── ...
        ├── 202598.jpg
        └── 202599.jpg
```





## CelebA-HQ (WIP)

### Brief description

### Download

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





## LFW (WIP)

### Brief description

[ **Labeled Faces in the Wild (LFW)**](http://vis-www.cs.umass.edu/lfw/) is a database of face photographs designed for studying the problem of unconstrained face recognition. The data set contains **more than 13,000 images of faces collected from the web**. Each face has been **labeled with the name of the person pictured**. **1680 of the people pictured have two or more distinct photos in the data set.** The only constraint on these faces is that they were detected by the Viola-Jones face detector. 

There are now four different sets of LFW images including the original and three different types of "aligned" images. The aligned images include "funneled images" (ICCV 2007), LFW-a, which uses an unpublished method of alignment, and "deep funneled" images (NIPS 2012). Among these, LFW-a and the deep funneled images produce superior results  for most face verification  algorithms over the original images and over the funneled images (ICCV 2007). 

### Download

Download the dataset following the instructions given [here](http://vis-www.cs.umass.edu/lfw/). The root directory of the dataset (`dataset_root`) should have the following structure:

http://vis-www.cs.umass.edu/lfw/lfw.tgz

http://vis-www.cs.umass.edu/lfw/lfw-names.txt

```bash
tar -xzvf lfw.tgz
```





TODO

```
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

