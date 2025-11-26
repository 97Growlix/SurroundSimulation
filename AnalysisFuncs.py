import numpy as np
import meshio
import felupe as fe
import gmsh
import pyvista as pv
import matplotlib.pyplot as plt
from felupe.constitution.tensortrax.models.hyperelastic import mooney_rivlin
from SurroundClasses import *


def AnalyzeItBothWays(NOPs):
    KmsOut, DispOut, field = AnalyzeItOneWay(NOPs)

    KmsIn, DispIn, field = AnalyzeItOneWay(NOPs, fe_mesh_field=field)
    
    KmsOut = KmsOut[1:] #remove the first element bc it gets repeated
    DispOut = DispOut[1:] 

    #add both sides of stroke together
    KmsTot = np.concatenate([KmsIn[::-1], KmsOut])
    DispTot = np.concatenate([DispIn[::-1], DispOut])

    return KmsTot, DispTot

def LocalRF(job, mask):  #finds the rf for masked nodes at the last step of the job
        Final_step_forces = job.steps[-1].items[0].results.force
        force_arr = Final_step_forces.toarray().flatten()
        num_nodes = force_arr.size // 3
        force_reshaped = force_arr.reshape((num_nodes, 3))

        masked_rfs = force_reshaped[mask, 2]
        rf = sum(masked_rfs)

        return rf

def CreateFeField(NOPs):

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)

    gmsh.open(NOPs.stepout_path)

    Refinement_pt_a = [NOPs.ConeWidth/2 - NOPs.ConeCornerRadius, NOPs.ConeHeight/2- NOPs.ConeCornerRadius, NOPs.ConeOffset]
    #Refinement_pt_b = [NOPs.ConeWidth, NOPs.ConeHeight - NOPs.ConeCornerRadius, NOPs.ConeOffset]

    dist_field = gmsh.model.mesh.field.add("Distance")

    pt_tag = gmsh.model.geo.addPoint(*Refinement_pt_a, 0)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.field.setNumbers(dist_field, "NodesList", [pt_tag])

    th_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(th_field, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(th_field, "SizeMin", NOPs.MeshFine)
    gmsh.model.mesh.field.setNumber(th_field, "SizeMax", NOPs.MeshCoarse)
    gmsh.model.mesh.field.setNumber(th_field, "DistMin", NOPs.ConeCornerRadius*1.1)
    gmsh.model.mesh.field.setNumber(th_field, "DistMax", NOPs.ConeCornerRadius*2)

    gmsh.model.mesh.field.setAsBackgroundMesh(th_field)

    gmsh.model.mesh.generate(3)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # version 2.2
    gmsh.option.setNumber("Mesh.Binary", 0)            # ASCII, not binary

    gmsh.write("surround.msh")

    #gmsh.fltk.run()
    #gmsh.finalize()

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

    return field.copy(), tetra_mesh, tetra_cells, points

def CreateBCs(NOPs, field, points):
    #masking
    #Inner Curved Surface
    Arc_Center_X = NOPs.ConeWidth/2 - NOPs.ConeCornerRadius
    Arc_Center_Y = NOPs.ConeHeight/2 - NOPs.ConeCornerRadius
    Z_Height = NOPs.ConeOffset
    tol = NOPs.Node_find_tol

    x_mod = points[:,0] - Arc_Center_X
    y_mod = points[:,1] - Arc_Center_Y
    z = points[:,2]

    r = np.sqrt(x_mod**2 + y_mod**2)

    R_outer_for_inner = NOPs.ConeCornerRadius+tol
    R_inner_for_outer = NOPs.ConeCornerRadius+NOPs.ConeEnclosureGap

    Inner_mask = (np.abs(z - Z_Height) <= tol) &\
        (r<=R_outer_for_inner) &\
        (x_mod>=-tol) & (y_mod>=-tol)
    
    #InnerTop
    Inner_top_mask = (np.abs(z - Z_Height) <= tol) &\
        (x_mod <= tol) & (y_mod <= NOPs.ConeCornerRadius + tol)

    #InnerRight
    Inner_right_mask = (np.abs(z - Z_Height) <= tol) &\
        (y_mod <= tol) &\
        (x_mod <= NOPs.ConeCornerRadius + tol)

    #Outer anchored surface

    Outer_anchored_mask = ((np.abs(z) <= tol)) &\
        (((r>=R_inner_for_outer) & (x_mod >= -tol) & (y_mod >= -tol)) |\
        ((y_mod>=NOPs.ConeCornerRadius + NOPs.ConeEnclosureGap) | (x_mod >= NOPs.ConeCornerRadius + NOPs.ConeEnclosureGap)))



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
        mask = Outer_anchored_mask,             # select points where z ≈ 0
        skip = (False, False, False)
    )
    #Create BC for applying displacements later
    move_z = fe.Boundary(
        field[0],
        mask = (Inner_mask | Inner_top_mask | Inner_right_mask),            # selects all nodes from the earlier mask
        skip=(True, True, False) # constrain only Z (doesn't actually matter I think)
    )
    bc_cone = fe.Boundary(
        field[0],
        mask = (Inner_mask | Inner_top_mask | Inner_right_mask),            # selects all nodes from the earlier mask
        skip=(False, False, True) # constrain only Z (doesn't actually matter I think)
    )

    #group all the bcs together
    bcs = dict()
    bcs["symx"] = bc_symx
    bcs["symy"] = bc_symy
    bcs["ground"] = bc_bottom
    bcs["move_z"] = move_z
    bcs["bc_cone"] = bc_cone

    masks = dict()
    masks["InnerMask"] = Inner_mask
    masks["InnerTop"] = Inner_top_mask
    masks["InnerRight"] = Inner_right_mask
    masks["Outer"] = Outer_anchored_mask


    return bcs, masks

def ShowItOff(NOPs):
    import os
    import subprocess
    import imageio

    field, tetra_mesh, tetra_cells, points = CreateFeField(NOPs)

    bcs, masks = CreateBCs(NOPs, field, points)
    #print("FElupe mesh ready:")
    print("Number of points in mesh:", tetra_mesh.points.shape[0])
    #print("Number of tetrahedra:", tetra_cells.shape[0])
    #print("Quadrature:", tetra_mesh.quadrature)

    #pyvista visualization

    # PyVista requires each cell to start with the number of points in the cell (4 for tetra)
    cells_pv = np.hstack([np.full((tetra_cells.shape[0], 1), 4), tetra_cells]).flatten()

    #format the data for pyvista
    grid = pv.UnstructuredGrid(cells_pv, np.full(tetra_cells.shape[0], 10), points)


    pv_plotter = pv.Plotter()
    pv_plotter.add_mesh(grid, show_edges = True)

    #function for adding glyphs to show BCs
    def add_bc_nodes(plotter, nodes, color):
        nodes_cloud = pv.PolyData(points[nodes])
        # glyph spheres with radius 1% of bounding box size
        bbox = grid.bounds
        radius = 0.005 * max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
        glyphs = nodes_cloud.glyph(scale=False, geom=pv.Sphere(radius=radius), orient = False)
        pv_plotter.add_mesh(glyphs, color=color)

    add_bc_nodes(pv_plotter, bcs["symx"].points, 'red')
    add_bc_nodes(pv_plotter, bcs["symy"].points, 'green')
    add_bc_nodes(pv_plotter, bcs["ground"].points, 'blue')
    add_bc_nodes(pv_plotter, bcs["move_z"].points, 'orange')


    ##unconnemt this to show boundary conds
    pv_plotter.camera_position = [(-125.0,-125.0,-25),(50.0,75.0,7.0),(0.0,0.0,1.0)]
    
    pv_plotter.show()


    ###
    #material data
    ###

    material = fe.Hyperelastic(mooney_rivlin, C10 = NOPs.MaterialCoefficients[0], C01 = NOPs.MaterialCoefficients[1])


    # Define your solid body
    solid = fe.SolidBody(material, field)

    # Define the displacement “ramp” (0 → some value) in linsteps

    steps_out = fe.math.linsteps([0,NOPs.Xmax], num=round(NOPs.N_Steps/2))
    steps_in = fe.math.linsteps([0,-NOPs.Xmax], num=round(NOPs.N_Steps/2))


    # Define a step
    step_out = fe.Step(
        items=[solid],
        ramp={bcs["move_z"]: steps_out},
        boundaries=bcs  # includes your fixed BCs + the “move_z” boundary
    )

    # Define a step
    step_in = fe.Step(
        items=[solid],
        ramp={bcs["move_z"]: steps_in},
        boundaries=bcs  # includes your fixed BCs + the “move_z” boundary
    )

    # Precompute PyVista cells
    cells_pv = np.hstack([np.full((tetra_cells.shape[0], 1), 4), tetra_cells]).flatten()
    cell_types = np.full(tetra_cells.shape[0], 10)  # 10 = VTK_TETRA

    original_pts = tetra_mesh.points.copy()


    deformed_meshes = []

    #Callback funtion to grab the deformed meshes at each step
    def cbout(stepnum, substepnum, substep, **kwargs):
        # Extract displacement field as NumPy array
        disp_values = substep.x[0].values  # convert Field → ndarray
        pts_def = original_pts + disp_values
        
        # Create a PyVista UnstructuredGrid for this timestep
        mesh_pv = pv.UnstructuredGrid(cells_pv, cell_types, pts_def)
        deformed_meshes.append(mesh_pv)

    def cbin(stepnum, substepnum, substep, **kwargs):
        # Extract displacement field as NumPy array
        disp_values = substep.x[0].values  # convert Field → ndarray
        pts_def = original_pts + disp_values
        
        # Create a PyVista UnstructuredGrid for this timestep
        mesh_pv = pv.UnstructuredGrid(cells_pv, cell_types, pts_def)
        deformed_meshes.insert(0,mesh_pv)


    ### Create the CharacteristicCurve job
    job_out = fe.CharacteristicCurve(steps=[step_out], boundary=bcs["move_z"], callback = cbout)
    job_in = fe.CharacteristicCurve(steps=[step_in], boundary=bcs["move_z"], callback = cbin)
    

    # Evaluate the characteristic curve
    job_out.evaluate()
    job_in.evaluate()

    ### For animation
    gif_path = "surround_mesh_animation.gif"

    # Create plotter window (visible)
    pl = pv.Plotter(window_size=(900, 700))
    pl.camera_position = [(NOPs.ConeWidth/2,-125.0, 7.0),(NOPs.ConeWidth/2,0, 7.0),(0.0,0.0,1.0)]
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
    imageio.mimsave("surround_mesh_animation.gif", frames + frames[::-1], fps=4, loop = 0)
    print("GIF saved:", gif_path)

    #open the gif automatically
    try:
        os.startfile(gif_path) #windows
    except AttributeError:
        subprocess.call(["open", gif_path]) #mac or Linux
        

    # Extract data I want
    #split rfs between straight and curved sections
    CurveRf = LocalRF(job_out, masks["InnerMask"])
    InnerTopRF = LocalRF(job_out, masks["InnerTop"])
    InnerRightRF = LocalRF(job_out, masks["InnerRight"])

    Angle_to_top_corner = np.arctan2(NOPs.ConeHeight/2+NOPs.ConeCornerRadius, NOPs.ConeWidth/2)  #abs angle to top starting part of arc
    Angle_to_bottom_corner = np.arctan2(NOPs.ConeHeight/2, NOPs.ConeWidth/2+NOPs.ConeCornerRadius)#abs angle to bottom point of arc
    
    Angular_span_curve = np.degrees(Angle_to_top_corner-Angle_to_bottom_corner)
    Rf_per_deg_curve = CurveRf/Angular_span_curve

    Angular_span_straight = np.degrees(Angle_to_bottom_corner + (np.pi/2-Angle_to_top_corner))
    Rf_per_deg_straight = (InnerTopRF + InnerRightRF)/Angular_span_straight

    print('Reaction force in Newtons per degree (curve vs flat section)')
    print(Rf_per_deg_curve)
    print(Rf_per_deg_straight)

    disp_out = np.array([d[2] for d in job_out.x])      # Z displacement
    force_out = 4*np.array([f[2] for f in job_out.y])     # Z reaction force (total)
    disp_in = np.array([d[2] for d in job_in.x])      # Z displacement
    force_in = 4*np.array([f[2] for f in job_in.y])     # Z reaction force (total)
    
    disp_in = disp_in[1:]
    force_in = force_in[1:]

    force = np.concatenate([force_in[::-1], force_out])
    disp = np.concatenate([disp_in[::-1], disp_out])
    
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

    return 0 

def AnalyzeItOneWay(NOPs, fe_mesh_field = None):
   
    ###
    # Moony-Rivlin model params
    ###
    
    C10 = NOPs.MaterialCoefficients[0]
    C01 = NOPs.MaterialCoefficients[1]

    if fe_mesh_field == None:
        field, tetra_mesh, tetra_cells, points = CreateFeField(NOPs)
    
    else:
        field = fe_mesh_field
        tetra_mesh = field[0].region.mesh
        points = tetra_mesh.points

    bcs, masks = CreateBCs(NOPs, field, points)
   
    ###
    #material data
    ###

    material = fe.Hyperelastic(mooney_rivlin, C10=C10, C01=C01)
    

    # Define your solid body
    solid = fe.SolidBody(material, field)
    
    # Define the displacement “ramp” (0 → some value) in linsteps

    steps = fe.math.linsteps([0,NOPs.Xmax], num=round(NOPs.N_Steps/2))
    
    # Define a step
    step = fe.Step(
        items=[solid],
        ramp={bcs["move_z"]: steps},
        boundaries=bcs  # includes your fixed BCs + the “move_z” boundary
    )

    ### Create the CharacteristicCurve job
    job = fe.CharacteristicCurve(steps=[step], boundary=bcs["move_z"])
    # Evaluate the characteristic curve

    job.evaluate()#filename=xdmf_filename)

    # Extract data useful for optimization
    disp = np.array([d[2] for d in job.x])      # Z displacement
    force = 4*np.array([f[2] for f in job.y])     # Z reaction force (total)

    # Compute stiffness
    kms = np.gradient(force, disp)

    return kms, disp, fe_mesh_field


def main():

  
    return 0 

if __name__ == "__main__":
    main()
