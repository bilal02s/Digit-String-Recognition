import numpy as np

one = np.ones((2, 2))

vertical = np.array([[1, 0, -1],
                     [1, 0, -1],
                     [1, 0, -1]])

horizontal = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1,-1,-1]])

diagonal = np.array([[2, 1, 0],
                     [1, 0,-1],
                     [0,-1, 2]])

gaussian = np.array([[1/4, 2/4, 1/4],
                     [2/4, 4/4, 2/4],
                     [1/4, 2/4, 1/4]])
                     
blur = np.array([[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]])