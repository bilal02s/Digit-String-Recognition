def modify_data(matrices):
    modified = []

    for matrix in matrices:
        noise = np.random.randn(28, 28)*0.1*255
        final = np.clip(matrix + noise, 0, 255)
        modified.append(final)

    return modified

def modify_data(matrices):
    modified = []

    for matrix in matrices:
        noise = np.random.randn(28, 28)*randint(0, 120)/1000*255
        final = np.clip(matrix + noise, 0, 255)
        noise = np.random.randn(28, 28)*randint(0, 120)/1000*255
        image2 = np.clip(matrix + noise, 0, 255)
        rotated = np.array(Image.fromarray(np.uint8(final)).rotate(randint(-15, 45)))
        modified.append(final)
        modified.append(rotated)

    return modified
    