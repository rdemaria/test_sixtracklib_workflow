import pickle
with open('line.pkl', 'rb') as fid:
    line = pickle.load(fid)

import simpletrack
elements = simpletrack.Elements()

for name, etype, ele in line:
    getattr(elements, etype)(**ele.as_dict())
