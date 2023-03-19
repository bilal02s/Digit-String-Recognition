import numpy as np

THRESHOLD = 80
NEIGHBOUR_THRESHOLD = 40

def filterMatrix(matrix):
    matrix = matrix.copy()
    all_cpnts = ConnectedComponentLabeling(matrix)

    #we will chose the biggest component
    size = [0 for cpnt in all_cpnts]
    for i in range(len(size)):
        size[i] = len(all_cpnts[i][0])

    keep = np.argmax(size)

    for i in range(len(all_cpnts)):
        if i == keep:
            continue 

        Is, Js = all_cpnts[i]
        matrix[Is, Js] = 0

    return matrix

def ConnectedComponentLabeling(matrix):
    row, col = matrix.shape 
    visited = np.zeros((row, col))
    all_components = []

    for i in range(row):
        for j in range(col):
            if matrix[i][j] < THRESHOLD or visited[i][j]:
                continue 
            
            cpnt = ([i], [j])
            explore(matrix, cpnt, visited)
            all_components.append(cpnt)

    return all_components

def explore(matrix, cpnts, visited):
    start = (cpnts[0].pop(0), cpnts[1].pop(0))
    unvisited = [start]

    while(len(unvisited) != 0):
        current = unvisited.pop(0)
        i, j = current 
        visited[i][j] = 1
        cpnts[0].append(i)
        cpnts[1].append(j)

        for In, Jn in neighbours(current, matrix.shape):
            if (not visited[In][Jn]) and ((In, Jn) not in unvisited) and (matrix[In][Jn] > NEIGHBOUR_THRESHOLD):
                unvisited.append((In, Jn))

def neighbours(pos, shape):
    row, col = shape 
    x, y = pos 
    neighbours = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            if x+i<0 or x+i>=row or y+j<0 or y+j>=col or i==j==0:
                continue 
            
            neighbours.append((x+i, y+j))

    return neighbours