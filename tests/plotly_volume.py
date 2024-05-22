import itertools
import numpy as np
import matplotlib.pyplot as plt
from nsfds3.utils import get_objects
from nsfds3.utils.data import Hdf5Wrapper
import plotly.graph_objects as go


cfg, msh = get_objects('configurations/data/', 'curvi_3d')
with Hdf5Wrapper(cfg.files.data_path) as file:
    p3d = file.get('p', iteration=0)



p3d = np.array(p3d)
x, y, z = np.meshgrid(msh.x, msh.y, msh.z)




# fig = go.Figure(data=go.Volume(
#     x=x.flatten(),
#     y=y.flatten(),
#     z=z.flatten(),
#     value=p3d.flatten() / p3d.max(),
#     isomin=-1,
#     isomax=1,
#     opacity=0.1, # needs to be small to see through all surfaces
#     surface_count=20, # needs to be a large number for good volume rendering
#     ))
# fig.show()

from mayavi import mlab
import vtk


data = np.zeros(msh.xp.shape)
for o in msh.obstacles:
    data[*o.sn] = 1

f = mlab.figure()
f.scene.movie_maker.record = True

mlab.contour3d(msh.xp, msh.yp, msh.zp, data, color=(0, 0, 0))

#s = mlab.contour3d(msh.xp, msh.yp, msh.zp, p3d, contours=5, transparent=True)
s = mlab.pipeline.volume(mlab.pipeline.scalar_field(msh.xp, msh.yp, msh.zp, p3d))


@mlab.animate(delay=250, ui=False)
@mlab.show()
def anim():
    for i in range(0, cfg.sol.nt, cfg.sol.ns):
        with Hdf5Wrapper(cfg.files.data_path) as file:
            p3d = file.get('p', iteration=i)
            s.mlab_source.scalars = p3d
            yield

anim()
