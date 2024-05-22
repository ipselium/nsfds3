import numpy as np
from nsfds3.utils import get_objects
from nsfds3.utils.data import Hdf5Wrapper
from mayavi import mlab
import vtk
from tvtk.tools import visual


cfg, msh = get_objects('configurations/data/', 'curvi_3d')
with Hdf5Wrapper(cfg.files.data_path) as file:
    p3d = np.array(file.get('p', iteration=0))

fig = mlab.figure(size=(800,800))
fig.scene.movie_maker.record = True
fig.scene.camera.azimuth(120)
fig.scene.camera.elevation(-20)
visual.set_viewer(fig)

obstacles = np.zeros(msh.xp.shape)
for o in msh.obstacles:
    obstacles[*o.sn] = 1

o = mlab.contour3d(msh.xp, msh.yp, msh.zp, obstacles, color=(0, 0, 0), figure=fig)
s = mlab.contour3d(msh.xp, msh.yp, msh.zp, p3d, contours=5, transparent=True, opacity=0.8, figure=fig)
#s = mlab.pipeline.volume(mlab.pipeline.scalar_field(msh.xp, msh.yp, msh.zp, p3d, figure=fig))


@mlab.show
@mlab.animate(delay=100, ui=True)
def anim():
    for i in range(cfg.sol.ns, cfg.sol.nt, cfg.sol.ns):
        print('Updating scene...', i)
        with Hdf5Wrapper(cfg.files.data_path) as file:
            s.mlab_source.scalars = file.get('p', iteration=i)
        yield

anim()