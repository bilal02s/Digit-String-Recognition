1. network trained on MNIST original data, tested on unprocessed images.
    * **Structure:** conv (4 kernel) > MaxPooling > 4\*14\*14 > 300 > 120 > 25 > 10
    * **Kernels:** vertical, horizontal, diagonal, opposite diagonal
    * **Results:** 
        * digit 0, accuracy : 0.587
        * digit 1, accuracy : 0.282
        * digit 2, accuracy : 0.309
        * digit 3, accuracy : 0.481
        * digit 4, accuracy : 0.395
        * digit 5, accuracy : 0.464
        * digit 6, accuracy : 0.084
        * digit 7, accuracy : 0.261
        * digit 8, accuracy : 0.502
        * digit 9, accuracy : 0.088

2. network trained on MNIST original data, tested on unprocessed images.
    * **Structure:** conv (4 kernel) > MaxPooling > conv (3 kernel) > MaxPooling > 12\*7\*7 > 250 > 50 > 10
    * **Kernels:** 
        * ConvLayer 1: vertical, horizontal, diagonal, opposite diagonal
        * ConvLayer 2: vertical, horizontal, diagonal
    * **Results:**
        * digit 0, accuracy : 0.448
        * digit 1, accuracy : 0.47
        * digit 2, accuracy : 0.41
        * digit 3, accuracy : 0.235
        * digit 4, accuracy : 0.817
        * digit 5, accuracy : 0.354
        * digit 6, accuracy : 0.382
        * digit 7, accuracy : 0.352
        * digit 8, accuracy : 0.431
        * digit 9, accuracy : 0.018

3. network trained on MNIST original data, tested on images processed slightly.
    * **Structure:** conv (4 kernel) > MaxPooling > conv (3 kernel) > MaxPooling > 12\*7\*7 > 250 > 50 > 10
    * **Kernels:** 
        * ConvLayer 1: vertical, horizontal, diagonal, opposite diagonal
        * ConvLayer 2: vertical, horizontal, diagonal
    * **Test image (DIDA) processing:** removing background light (background pixels => 0)
    * **Results:**
        * digit 0, accuracy : 0.488
        * digit 1, accuracy : 0.476
        * digit 2, accuracy : 0.432
        * digit 3, accuracy : 0.246
        * digit 4, accuracy : 0.799
        * digit 5, accuracy : 0.344
        * digit 6, accuracy : 0.361
        * digit 7, accuracy : 0.355
        * digit 8, accuracy : 0.458
        * digit 9, accuracy : 0.017

4. network trained on MNIST images with different noise levels added.
    * **Train image processing:**
        * Noise: gaussian noise, mean=0, standard deviation=25
    * **Structure:** conv (4 kernel) > MaxPooling > conv (3 kernel) > MaxPooling > 12\*7\*7 > 250 > 50 > 10
    * **Kernels:** 
        * ConvLayer 1: vertical, horizontal, diagonal, opposite diagonal
        * ConvLayer 2: vertical, horizontal, diagonal
    * **Test image (DIDA) processing:** the same as the step before.
    * **Results:**
        * digit 0, accuracy : 0.584
        * digit 1, accuracy : 0.578
        * digit 2, accuracy : 0.305
        * digit 3, accuracy : 0.252
        * digit 4, accuracy : 0.702
        * digit 5, accuracy : 0.307
        * digit 6, accuracy : 0.255
        * digit 7, accuracy : 0.397
        * digit 8, accuracy : 0.502
        * digit 9, accuracy : 0.041

5. network trained on MNIST images rotated at different angles and different noise levels
    * **Train image processing:**
        * Rotation: random rotation between -15 and 45 degree counter clockwise
        * Noise: gaussian noise, mean=0, standard deviation=25
    * **Results on MNIST test images:**
        * accuracy : 0.955
    * **Structure:** conv (4 kernel) > MaxPooling > conv (3 kernel) > MaxPooling > 12\*7\*7 > 250 > 50 > 10
    * **Kernels:** 
        * ConvLayer 1: vertical, horizontal, diagonal, opposite diagonal
        * ConvLayer 2: vertical, horizontal, diagonal
    * **Test image (DIDA) processing:** the same as the step before.
    * **Results:**
        * digit 0, accuracy : 0.512
        * digit 1, accuracy : 0.667
        * digit 2, accuracy : 0.494
        * digit 3, accuracy : 0.279
        * digit 4, accuracy : 0.599
        * digit 5, accuracy : 0.399
        * digit 6, accuracy : 0.173
        * digit 7, accuracy : 0.296
        * digit 8, accuracy : 0.461
        * digit 9, accuracy : 0.057
    * **Confusion matrix:**  <br/>
        [[512   6   5 106   3  50  93   0  13  16] <br/>
        [ 45 667  40  18  88 142 177 171  63  90] <br/>
        [205 131 494 311 150  70 105 388 134 269] <br/>
        [  5   1  25 279   9   6  30   9  35  14] <br/>
        [ 42  72  44  25 599  29  41  45  99 196] <br/>
        [  8  14  11  48   8 399 245  22 101  18] <br/>
        [ 21   5  11  23   7  16 173   2  32  15] <br/>
        [114  61  14  50  74 136  48 296  28 259] <br/>
        [  5  25 337  97  40 109  50  53 461  66] <br/>
        [ 43  18  19  43  22  43  38  14  34  57]] <br/>

    * Same parameters as before, except the rotation angle of training images: between -40 and 15 counter clockwise <br/>
    * **Results:**
        * digit 0, accuracy : 0.509
        * digit 1, accuracy : 0.578
        * digit 2, accuracy : 0.476
        * digit 3, accuracy : 0.267
        * digit 4, accuracy : 0.726
        * digit 5, accuracy : 0.385
        * digit 6, accuracy : 0.23
        * digit 7, accuracy : 0.503
        * digit 8, accuracy : 0.501
        * digit 9, accuracy : 0.038
    * **Confusion matrix:** <br/>
        [[509   9  15  58   2  10  90   3  14   8] <br/>
        [ 32 578  24   7  63  71  74 110  33  60] <br/>
        [109 120 476 202  77 100 115 244  67 121] <br/>
        [  1   0  18 267   2  12   8  13  25   8] <br/>
        [ 96 130  80  48 726  57  96  73 192 189] <br/>
        [  4   1   9  27   3 385 158   9  62   7] <br/>
        [ 12   6   7  10   1  12 230   1  14   1] <br/>
        [202 123  50 194  95 177 121 503  53 527] <br/>
        [ 14  29 308 161  28 153  90  41 501  41] <br/> 
        [ 21   4  13  26   3  23  18   3  39  38]] <br/>

6. network trained on MNIST image modified with variety of techniques to simulate real world test images. Training set consists of Mnist original images, added to it the same images augmented using various modification. In total the train set size is double the Mnist size: 120k images.
    * **Train image processing:** 
        * Random zoom in/zoom out
        * Random rotation : between -30 and 15 degree counter-clockwise
        * Random translation : random vector between the two vectors (-5, -5) and (5, 5)
        * Random noise : gaussian noise, mean=0, standard deviation=(random value between 0 and 20)
        * Random scratchs : random scratchs (between 1 and 2) in the form of linear line with a random slope 
    * **Results on Mnist test images**:
        * Mnist original test images: accuracy=0.952
        * MNist augmented test images (20k test image): accuracy=0.83
    * **Structure:** conv (4 kernel) > MaxPooling > conv (3 kernel) > MaxPooling > 12\*7\*7 > 250 > 50 > 10
    * **Kernels:** 
        * ConvLayer 1: vertical, horizontal, diagonal, opposite diagonal
        * ConvLayer 2: vertical, horizontal, diagonal
    * **Test image (DIDA) processing:** same processing as the step before.
    * **Results:**
        * digit 0, accuracy : 0.737
        * digit 1, accuracy : 0.84
        * digit 2, accuracy : 0.42
        * digit 3, accuracy : 0.246
        * digit 4, accuracy : 0.805
        * digit 5, accuracy : 0.307
        * digit 6, accuracy : 0.58
        * digit 7, accuracy : 0.459
        * digit 8, accuracy : 0.538
        * digit 9, accuracy : 0.122
    * **Confusion matrix:** <br/>
        [[737  14  39 140   7  79  97  10  78  60] <br/>
        [ 99 840  25  32 126 143 155 174  41 103] <br/>
        [ 23  13 420 152  17  38  27  67  36  26] <br/>
        [  0   0  27 246   1  10   5  10   6  10] <br/>
        [ 88  79 126  63 805  94  76 165 152 411] <br/>
        [  0   0   5  46   0 307  26   7  41   0] <br/>
        [ 32  35  29  32  12  33 580   6  75   5] <br/>
        [ 12  11  13  48  19  40   5 459   9 225] <br/>
        [  8   8 292 185   8 212  22  60 538  38] <br/>
        [  1   0  24  56   5  44   7  42  24 122]] <br/>

7. network trained on MNIST image modified with variety of techniques to simulate real world test images.
    * **Train image processing:** the same training set as the step before
    * **Structure:** conv (4 kernel) > MaxPooling > conv (3 kernel) > MaxPooling > 12\*7\*7 > 250 > 50 > 10
    * **Kernels:** 
        * ConvLayer 1: vertical, horizontal, diagonal, opposite diagonal
        * ConvLayer 2: vertical, horizontal, diagonal
    * **Test image (DIDA) processing:** 
        * Same processing as the step before (rescaling intensities between 0 and 255, removing background light).
        * Filtering images :
            * by creating *Connected Component Labeling*, as a result we get the list of coordinates of all the disconnected components in the image.
            * Keeping the components with the most number of pixels, (assuming that the number we are trying to find is the biggest component and all other components are noise), and then setting all other component's pixels to zero.
    * **Results:**
        * digit 0, accuracy : 0.751
        * digit 1, accuracy : 0.927
        * digit 2, accuracy : 0.376
        * digit 3, accuracy : 0.411
        * digit 4, accuracy : 0.761
        * digit 5, accuracy : 0.394
        * digit 6, accuracy : 0.589
        * digit 7, accuracy : 0.608
        * digit 8, accuracy : 0.683
        * digit 9, accuracy : 0.244
    * **Confusion matrix**: <br/>
        [[751   7  10  49   0  45  69   3  32  24] <br/>
        [125 927  35  38 177 154 174 193  40 151] <br/>
        [ 15   6 376  91   9  22  16  28  27  23] <br/>
        [  0   2  53 411   0  49   4   8  13  13] <br/>
        [ 24  22  76  41 761  29  33  62  40 172] <br/>
        [  0   2  13  45   2 394  56   7  85   4] <br/>
        [ 26  11  20   8   6  23 589   0  41   2] <br/>
        [ 28  15  22  62  33  59   5 608   7 322] <br/>
        [ 20   7 344 158   6 162  48  38 683  45] <br/>
        [ 11   1  51  97   6  63   6  53  32 244]] <br/>

8. network trained on MNIST image modified with variety of techniques to simulate real world test images.
    * **Train image processing:** the same training set as the step before
    * **Structure:** conv (8 kernel) > MaxPooling > conv (4 kernel) > MaxPooling > 32\*5\*5 > 250 > 80 > 10
    * **Padding:** Valid padding was used to help reduce further the dimentionality
    * **Kernels:** This network implemented backpropagation to learn its convolutinal layer's kernels
    * **Test image (DIDA) processing:** 
        * Same processing as the step before (rescaling intensities between 0 and 255, removing background light).
        * Filtering images :
            * by creating *Connected Component Labeling*, as a result we get the list of coordinates of all the disconnected components in the image.
            * Keeping the components with the most number of pixels, (assuming that the number we are trying to find is the biggest component and all other components are noise), and then setting all other component's pixels to zero.
    * **Results:**
        * digit 0, accuracy : 0.797
        * digit 1, accuracy : 0.928
        * digit 2, accuracy : 0.29
        * digit 3, accuracy : 0.565
        * digit 4, accuracy : 0.697
        * digit 5, accuracy : 0.517
        * digit 6, accuracy : 0.662
        * digit 7, accuracy : 0.702
        * digit 8, accuracy : 0.588
        * digit 9, accuracy : 0.38
        * overall accuracy : 0.6126
    * **Confusion matrix**: <br/>
        [[797  24  39  32   9  50  75   8  39  25] <br/>
         [150 928 113  91 208 224 164 184 101 183] <br/>
         [  3   4 290  39   7  43   6  37  35  10] <br/>
         [  1   1 141 565   0  30   0   8  31  18] <br/>
         [  7  12  50  22 697  20  36  13  54  80] <br/>
         [  5   2  29  43   5 517  44  14  46   8] <br/>
         [  5  12  22  12   8  18 662   3  28   1] <br/>
         [ 23  10  70  81  37  46   2 702  21 288] <br/>
         [  7   7 126  14  19  13   8  17 588   7] <br/>
         [  2   0 120 101  10  39   3  14  57 380]] <br/>

9. network trained on MNIST image modified with variety of techniques to simulate real world test images.
    * **Train image processing:** the same training set as the step before, + trainset image normalised to mean 0 and standard deviation of 1.
    * **Structure:** conv (8 kernel) > MaxPooling > conv (4 kernel) > MaxPooling > 12\*5\*5 > 250 > 80 > 10
    * **Padding:** Valid padding was used to help reduce further the dimentionality
    * **Kernels:** This network implemented backpropagation to learn its convolutinal layer's kernels
    * **Test image (DIDA) processing:** 
        * Same processing as the step before (rescaling intensities between 0 and 255, removing background light).
        * Normalising image's pixels to mean 0 and standard deviation of 1.
        * Filtering images :
            * by creating *Connected Component Labeling*, as a result we get the list of coordinates of all the disconnected components in the image.
            * Keeping the components with the most number of pixels, (assuming that the number we are trying to find is the biggest component and all other components are noise), and then setting all other component's pixels to zero.
    * **Results:**
        * digit 0, accuracy : 0.825
        * digit 1, accuracy : 0.898
        * digit 2, accuracy : 0.431
        * digit 3, accuracy : 0.57
        * digit 4, accuracy : 0.86
        * digit 5, accuracy : 0.502
        * digit 6, accuracy : 0.691
        * digit 7, accuracy : 0.732
        * digit 8, accuracy : 0.652
        * digit 9, accuracy : 0.395
        * overall accuracy : 0.6556
    * **Confusion matrix**: <br/>
        [[825  37  23  36   5  57 104  16  29  33] <br/>
         [109 898  42  45  89 112 101  91  41  85] <br/>
         [  6   4 431  75  12  47  22  57  31  12] <br/>
         [  1   0 161 570   4  40   1  11  13  30] <br/>
         [ 14  32  34  35 860  32  54  45  85 150] <br/>
         [  0   0  30  33   0 502  17   5  72   9] <br/>
         [ 14  13  37  10   1  40 691   2  21   3] <br/>
         [ 25   8  49  84  22  51   2 732  22 276] <br/>
         [  4   7 153  31   5  59   7  12 652   7] <br/>
         [  2   1  40  81   2  60   1  29  34 395]] <br/>

10. network trained on MNIST image modified with variety of techniques to simulate real world test images.
    * **Train image processing:** the same training set as the step before, except image resized to (42, 42) instead of (28, 28), + trainset image normalised to mean 0 and standard deviation of 1.
    * **Structure:** conv (8 kernel) > MaxPooling > conv (4 kernel) > MaxPooling > conv (4 kernel) > MaxPooling > 128\*3\*3 > 400 > 100 > 50 > 10
    * **Padding:** Valid padding was used to help reduce further the dimentionality
    * **Kernels:** This network implemented backpropagation to learn its convolutinal layer's kernels
    * **Test image (DIDA) processing:** 
        * Same processing as the step before (rescaling intensities between 0 and 255, removing background light).
        * Normalising image's pixels to mean 0 and standard deviation of 1.
        * Filtering images :
            * by creating *Connected Component Labeling*, as a result we get the list of coordinates of all the disconnected components in the image.
            * Keeping the components with the most number of pixels, (assuming that the number we are trying to find is the biggest component and all other components are noise), and then setting all other component's pixels to zero.
    * **Results:**
        * digit 0, accuracy : 0.863
        * digit 1, accuracy : 0.936
        * digit 2, accuracy : 0.517
        * digit 3, accuracy : 0.733
        * digit 4, accuracy : 0.796
        * digit 5, accuracy : 0.582
        * digit 6, accuracy : 0.754
        * digit 7, accuracy : 0.798
        * digit 8, accuracy : 0.764
        * digit 9, accuracy : 0.43
        * overall accuracy : 0.7172
    * **Confusion matrix**: <br/>
        [[863  16  14  21   2  20  52   2  20  23] <br/>
         [ 73 936  89  33 118 117  89 106  19  91] <br/>
         [  3   5 517  27  23  23  15  48  10  11] <br/>
         [  2   0 144 733   3  65   2  10  34  48] <br/>
         [ 17  17  29  10 796  29  29   9  46  72] <br/>
         [  3   0  31  49   0 582  38   1  57  23] <br/>
         [  4   6   6   6   3   8 754   0  12   2] <br/>
         [ 29  17  27  65  34  69   3 798   6 293] <br/>
         [  5   3 112  11  15  27  13   8 764   7] <br/>
         [  1   0  31  45   6  60   5  18  32 430]] <br/>