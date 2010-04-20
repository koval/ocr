#!/usr/bin/env python
import struct
import Image

labels = file('train-labels-idx1-ubyte', 'rb')
labels.seek(8)

images = file('train-images-idx3-ubyte', 'rb')
images.seek(16)

finished = 0
idx = {}

i = 0
while finished < 10:
    d = struct.unpack('B', labels.read(1))[0]
    l = idx.setdefault(d, [])
    if len(l) < 10:
        l.append(i)
        if len(l) == 10:
            finished += 1
    i += 1
from pprint import pprint
pprint(idx)

size = 28
img = Image.new('L', (size*10, size*10))
fragment = Image.new('L', (size, size))
for i in range(10):
    for j in range(10):
        images.seek(16+idx[j][i]*size*size)
        fragment.putdata(struct.unpack('B'*size*size, images.read(size*size)))
        img.paste(fragment, (j*size, i*size))
img = Image.eval(img, lambda p: 255-p)
img.save('digits.png', 'PNG')
