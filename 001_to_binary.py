import pickle
with open('line.pkl', 'rb') as fid:
    line = pickle.load(fid)

import pysixtracklib
elements = pysixtracklib.Elements.fromline(line)

for name, etype, ele in line:
    getattr(elements, etype)(**ele._asdict())

elements.tofile("elements.buffer")

elements.fromfile("elements.buffer")

