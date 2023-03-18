1. network trained on MNIST original data, tested on unprocessed images.
    * **Structure:** conv (4 kernel) > 4\*14\*14 > 300 > 120 > 25 > 10
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
    * **Structure:** conv (4 kernel) > conv (3 kernel) > 12\*7\*7 > 250 > 50 > 10
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
    * **Structure:** conv (4 kernel) > conv (3 kernel) > 12\*7\*7 > 250 > 50 > 10
    * **Test image processing:** removing background light (background pixels => 0)
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
    * **Structure:** conv (4 kernel) > conv (3 kernel) > 12\*7\*7 > 250 > 50 > 10
    * **Test image processing:** the same as the step before.
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
    * **Structure:** conv (4 kernel) > conv (3 kernel) > 12\*7\*7 > 250 > 50 > 10
    * **Test image processing:** the same as the step before.
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

6. network trained on MNIST image modified with variety of techniques to simulate real world test images.
    * **Train image processing:**
    * **Structure:** conv (4 kernel) > conv (3 kernel) > 12\*7\*7 > 250 > 50 > 10
    * **Test image processing:** same processing as the step before.
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

