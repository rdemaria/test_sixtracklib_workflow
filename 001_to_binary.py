import pickle
import numpy as np
import os

with open('line.pkl', 'rb') as fid:
    line = pickle.load(fid)

import pysixtracklib
elements = pysixtracklib.Elements.fromline(line)

for name, etype, ele in line:
    getattr(elements, etype)(**ele._asdict())

elements.tofile("elements.buffer")

ps = pysixtracklib.ParticlesSet()
p = ps.Particles(num_particles=100)
p.set_reference(p0c=450e9)
ps.tofile('particles.buffer')

os.system('../sixtracklib/build/examples/c99/track_io_c99 particles.buffer elements.buffer 10 1 1 1 >run.out')

res = pysixtracklib.ParticlesSet.fromfile('output_particles.bin')

# Test read
elements.fromfile("elements.buffer")
