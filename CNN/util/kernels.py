import numpy as np

one = np.ones((2, 2))

vertical = np.array([[1/3, 0, -1/3],
                     [1/3, 0, -1/3],
                     [1/3, 0, -1/3]])

horizontal = np.array([[1/3, 1/3, 1/3],
                       [0  ,   0,   0],
                       [-1/3,-1/3,-1/3]])

diagonal = np.array([[2/4, 1/4, 0],
                     [1/4, 0,-1/4],
                     [0,-1/4,-2/4]])

diagonal2 = np.array([[0, 1/4, 2/4],
                      [-1/4,0, 1/4],
                      [-2/4,-1/4,0]])

gaussian = np.array([[1/16, 2/16, 1/16],
                     [2/16, 4/16, 2/16],
                     [1/16, 2/16, 1/16]])
                     
blur = np.array([[1/9, 1/9, 1/9],
                 [1/9, 1/9, 1/9],
                 [1/9, 1/9, 1/9]])

gaussian_blur = np.array(
    [[1/273, 4/273, 7/273, 4/273, 1/273],
     [4/273, 16/273, 26/273, 16/273, 4/273],
     [7/273, 26/273, 41/273, 26/273, 7/273],
     [4/273, 16/273, 26/273, 16/273, 4/273],
     [1/273, 4/273, 7/273, 4/273, 1/273]]
)