import numpy as np
from PIL import Image
from PIL import ImageOps
from random import randint
from random import random
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN, GaussianBlur
)

class MyMatrix:
    def __init__(self, matrix):
        self.matrix = matrix.astype('float32')

    def getMatrix(self):
        return self.matrix.astype('float32')

    def addNoise(self, SD):
        noise = np.random.default_rng().normal(0, SD, size=self.matrix.shape)
        self.matrix = np.clip(self.matrix + noise, 0, 255) 
        return self

    def addRandomNoise(self, maxSD):
        return self.addNoise(randint(0, maxSD))

    def transform(self, angle=0, vector=None):
        self.matrix = np.array(Image.fromarray(np.uint8(self.matrix)).rotate(angle, translate=vector)).astype('float32')
        return self

    def randomTransformation(self, angleRange, vectorRange):
        minAngle, maxAngle = angleRange
        minVector, maxVector = vectorRange
        xmin, ymin = minVector
        xmax, ymax = maxVector 
        x, y = randint(xmin, xmax), randint(ymin, ymax)
        vector = (x, y)
        return self.transform(angle=randint(minAngle, maxAngle), vector=vector)

    def zoom(self, box):
        self.matrix = np.array(Image.fromarray(np.uint8(self.matrix)).crop(box).resize(self.matrix.shape)).astype('float32')
        return self

    def randomZoom(self, maxBox):
        maxleft, maxupper, maxright, maxbottom = maxBox 
        width, height = self.matrix.shape
        p = random()
        left, upper, right, bottom = p*maxleft, p*maxupper, width - (p*(width - maxright)), height - (p*(height - maxbottom))
        box = (left, upper, right, bottom)
        return self.zoom(box)

    def addRandomScratch(self, value):
        row, col = self.matrix.shape
        line_width = 1
        line = np.repeat(value, col*line_width).reshape(line_width, col) + np.random.randn(line_width, col)
        offset = randint(0, row-line_width) 
        scratch = np.pad(line, ((offset, row-offset-line_width), (0, 0)), mode='constant', constant_values=0)
        nb_scratch = randint(1, 2)

        gen = np.random.default_rng().normal
        for i in range(nb_scratch):
            im = MyMatrix(scratch.copy())
            angle = randint(0, 1)*gen(0, 10, size=1) + randint(0, 1)*gen(90, 10, size=1)
            im.transform(angle=angle)
            self.matrix = np.max((self.matrix, im.getMatrix()), axis=0)

        return self

    def blur(self):
        self.matrix = np.array(Image.fromarray(np.uint8(self.matrix)).filter(GaussianBlur(0.65))).astype('float32')
        return self


        
