import numpy as np
import os
import meshio
import felupe as fe
import gmsh
import pyvista as pv
import matplotlib.pyplot as plt
from felupe.constitution.tensortrax.models.hyperelastic import mooney_rivlin
import subprocess
import imageio


####
#User settings
####

surround_step = r"C:\Users\Gaming pc\Downloads\QuarterSurround.step"
mesh_size = 2 #in mm
Xmax = 5 #one way displacement
N_Steps = 5 #number of steps to calculate

###
# Moony-Rivlin model params
###

C10 = 1.908
C01 = 0.65

###
#Gmsh stuff
###

gmsh.initialize()
gmsh.open(surround_step)
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
gmsh.model.mesh.generate(3)

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # version 2.2
gmsh.option.setNumber("Mesh.Binary", 0)            # ASCII, not binary

gmsh.write("surround.msh")


###
#Visualization if you want to look at the mesh in gmsh's thing
###

#gmsh.fltk.run()
#gmsh.finalize()


####
#import mesh into felupe
####

# Read the Gmsh mesh
msh_file = "surround.msh"
msh = meshio.read(msh_file)

# Extract points
points = msh.points

# Extract tetrahedral cells
tetra_cells = None
for cell_block in msh.cells:
    if cell_block.type == "tetra":
        tetra_cells = cell_block.data
        break

if tetra_cells is None:
    raise ValueError("No tetrahedral mesh found in the file")

# Create the FElupe mesh
tetra_mesh = fe.Mesh(points, tetra_cells, cell_type = "tetra")

# Assign quadrature for tetrahedra
quad = fe.quadrature.Tetrahedron(order=2)
tetra_mesh.quadrature = quad
elem = fe.element.Tetra()
# Create a region over all tetra cells
region = fe.Region(tetra_mesh, elem, quadrature=quad)

# Create a field
u = fe.Field(region, dim=3)
field = fe.FieldContainer([u])


print("FElupe mesh ready:")
print("Number of points:", tetra_mesh.points.shape[0])
print("Number of tetrahedra:", tetra_cells.shape[0])
print("Quadrature:", tetra_mesh.quadrature)

###
#boundary cond
###

# Symmetry plane at x=0, fix only x displacement
bc_symx = fe.Boundary(
    field[0],
    fx=0.0,             # select points where x ≈ 0
    skip=(False, True, True)  # fix x, skip y and z
)

# Symmetry plane at y=0, fix only y displacement
bc_symy = fe.Boundary(
    field[0],
    fy=0.0,             # select points where y ≈ 0
    skip=(True, False, True)  # fix y, skip x and z
)

# Fixed support at z=0, fix all components
bc_bottom = fe.Boundary(
    field[0],
    fz=0.0,             # select points where z ≈ 0
    skip = (False, False, False)
)
#Create BC for applying displacements later
move_z = fe.Boundary(
    field[0],
    fz=12.0,            # selects all nodes at Z=12
    skip=(True, True, False) # constrain only Z (doesn't actually matter I think)
)

#group all the bcs together
bcs = dict()
bcs["symx"] = bc_symx
bcs["symy"] = bc_symy
bcs["ground"] = bc_bottom
bcs["move_z"] = move_z



#pyvista visualization

# PyVista requires each cell to start with the number of points in the cell (4 for tetra)
cells_pv = np.hstack([np.full((tetra_cells.shape[0], 1), 4), tetra_cells]).flatten()

#format the data for pyvista
points = msh.points
grid = pv.UnstructuredGrid(cells_pv, np.full(tetra_cells.shape[0], 10), points)


pv_plotter = pv.Plotter()
pv_plotter.add_mesh(grid, show_edges = True)

#function for adding glyphs to show BCs
def add_bc_nodes(plotter, nodes, color):
    nodes_cloud = pv.PolyData(points[nodes])
    # glyph spheres with radius 1% of bounding box size
    bbox = grid.bounds
    radius = 0.01 * max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
    glyphs = nodes_cloud.glyph(scale=False, geom=pv.Sphere(radius=radius))
    pv_plotter.add_mesh(glyphs, color=color)

add_bc_nodes(pv_plotter, bc_symx.points, 'red')
add_bc_nodes(pv_plotter, bc_symy.points, 'green')
add_bc_nodes(pv_plotter, bc_bottom.points, 'blue')
add_bc_nodes(pv_plotter, move_z.points, 'orange')


##unconnemt this to show boundary conds
#pv_plotter.show()


###
#material data
###

material = fe.Hyperelastic(mooney_rivlin, C10=C10, C01=C01)


# Define your solid body
solid = fe.SolidBody(material, field)

# Define the displacement “ramp” (0 → some value) in linsteps

steps = fe.math.linsteps([0,Xmax], num=N_Steps)

# Define a step
step = fe.Step(
    items=[solid],
    ramp={bcs["move_z"]: steps},
    boundaries=bcs  # includes your fixed BCs + the “move_z” boundary
)

# Precompute PyVista cells
cells_pv = np.hstack([np.full((tetra_cells.shape[0], 1), 4), tetra_cells]).flatten()
cell_types = np.full(tetra_cells.shape[0], 10)  # 10 = VTK_TETRA

original_pts = tetra_mesh.points.copy()


deformed_meshes = []

#Callback funtion to grab the deformed meshes at each step
def cb(stepnum, substepnum, substep, **kwargs):
    # Extract displacement field as NumPy array
    disp_values = substep.x[0].values  # convert Field → ndarray
    pts_def = original_pts + disp_values
    
    # Create a PyVista UnstructuredGrid for this timestep
    mesh_pv = pv.UnstructuredGrid(cells_pv, cell_types, pts_def)
    deformed_meshes.append(mesh_pv)




### Create the CharacteristicCurve job
job = fe.CharacteristicCurve(steps=[step], boundary=bcs["move_z"], callback = cb)


# Evaluate the characteristic curve
xdmf_filename = "surround_characteristic.xdmf"
job.evaluate(filename=xdmf_filename)




### For animation
gif_path = "surround_mesh_animation.gif"

# Create plotter window (visible)
pl = pv.Plotter(window_size=(900, 700))
pl.camera_position = [(-125.0,-125.0,-25),(50.0,75.0,7.0),(0.0,0.0,1.0)]
pl.add_mesh(deformed_meshes[0], show_edges=True)
pl.add_text("Mesh displacement animation", font_size=10)
cpos = pl.camera_position
pl.show(auto_close=False, interactive=False)


#loop through the meshes collected to create frames
frames = []
for t in range(len(deformed_meshes)):
    # Update the mesh points of the same PyVista object
    pl.mesh.points[:] = deformed_meshes[t].points  # or replace with new mesh
    pl.render()
    frames.append(pl.screenshot(return_img=True))

pl.close()
imageio.mimsave("surround_mesh_animation.gif", frames, fps=4, loop = 0)
print("GIF saved:", gif_path)

#open the gif automatically
try:
    os.startfile(gif_path) #windows
except AttributeError:
    subprocess.call(["open", gif_path]) #mac or Linux
    

# Extract data useful for optimization
disp = np.array([d[2] for d in job.x])      # Z displacement
force = 4*np.array([f[2] for f in job.y])     # Z reaction force (total)

# Compute stiffness
kms = np.gradient(force, disp)


##plot the graphs of force and kms
fig, axs = plt.subplots(2, 1, figsize=(6, 10))

axs[0].plot(disp, force, '-o')
axs[0].set_xlabel("Displacement [mm]")
axs[0].set_ylabel("Reaction Force [N]")
axs[0].set_title("Reaction Force")
axs[0].grid(True)

axs[1].plot(disp, kms, '-o')
axs[1].set_xlabel("Displacement [mm]")
axs[1].set_ylabel("Stiffness [N/mm]")
axs[1].set_title("Kms")
axs[1].grid(True)
axs[1].set_ylim([0,max(kms)*1.3])

plt.tight_layout()
plt.show()
