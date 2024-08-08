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


