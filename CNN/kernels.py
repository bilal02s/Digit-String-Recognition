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

gaussian = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]])
                     
blur = np.array([[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]])