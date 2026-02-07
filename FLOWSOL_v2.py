import numpy as np
import matplotlib.pyplot as plt
import time
from colorama import Fore, Style, init
from matplotlib.animation import FuncAnimation
import psutil
from scipy.sparse.linalg import cg
from tqdm import tqdm
import torch
import scipy
import os
import cupyx.scipy.sparse as gpu_sp
import cupy as cp
import cupyx.scipy.sparse.linalg as spla

output_dir = r"D:/numerical computation/Results/FAH4001"

os.makedirs(output_dir, exist_ok=True)
#=======================================================================================================================================#
#                                                                VERSION CHECK BLOCK
#=======================================================================================================================================#
print("="*60)
print("🔎 Environment & Library Versions")
print("="*60)

import sys

print("Python version        :", sys.version.replace("\n", " "))

print("NumPy version         :", np.__version__)
print("Matplotlib version    :", plt.matplotlib.__version__)
print("SciPy version         :", scipy.__version__)
print("CuPy version          :", cp.__version__)
print("PyTorch version       :", torch.__version__)
print("psutil version        :", psutil.__version__)

# CUDA info (PyTorch)
print("\n 🖥️ CUDA / GPU Info (PyTorch)")
print("CUDA available        :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version          :", torch.version.cuda)
    print("GPU name              :", torch.cuda.get_device_name(0))
    print("GPU capability        :", torch.cuda.get_device_capability(0))

# CUDA info (CuPy)
print("\n 🖥️ CUDA / GPU Info (CuPy)")
try:
    print("CuPy CUDA runtime     :", cp.cuda.runtime.runtimeGetVersion())
    print("CuPy GPU name         :", cp.cuda.Device(0).name)
except Exception as e:
    print("CuPy CUDA info error  :", e)

print("="*60)

# Total and available memory (in GB)
total = psutil.virtual_memory().total / (1024**3)
available = psutil.virtual_memory().available / (1024**3)
print(Fore.GREEN + f"total space {total} GB" + Style.RESET_ALL)
print(Fore.RED + f"available space {available} GB" + Style.RESET_ALL)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}") 
#=======================================================================================================================================#
#                                                                       END
#=======================================================================================================================================#

#=======================================================================================================================================#
#                                                                   IMPORTING DATA
#=======================================================================================================================================#

mesh_data = np.load(r"D:/numerical computation/geometry meshing/Meshes/RAX_1.npz")
gax_file = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
# sorted_first_interface = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
# first_interface = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
# second_interface = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
inside_pt = mesh_data["array1"]
ghost_nodes = gax_file["array1"]
sorted_first_interface = gax_file["array7"]
sfi = gax_file["array8"]
ffi = gax_file["array9"]
ghost_nodes_list = ghost_nodes_list = [list(map(tuple, block)) for block in ghost_nodes]
first_interface = set(tuple(point) for point in ffi)
first_interface = np.array(
    sorted(first_interface),
    dtype=float)

second_interface = np.array(list(sfi.item()), dtype=float)
del_h = float(mesh_data["del_h"]) 

# print(">/\<",sfi[0])
#=======================================================================================================================================#
#                                                                       END
#=======================================================================================================================================#

#=======================================================================================================================================#
#                                                                   REBUILDING DOMAIN
#=======================================================================================================================================#

conversion_factor = 1/del_h     # mesh size
# conversion_factor = 
print("grid size ",inside_pt[0][0],inside_pt[0][1])
print(conversion_factor)


def cord_transfer_logic(a):
    x_coord=inside_pt[a][0]
    y_coord=inside_pt[a][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    return r,c 

def cord_transfer_logic_l1(a):
    x_coord=first_interface[a][0]
    y_coord=first_interface[a][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    return r,c 

def cord_transfer_logic_l2(a):
    x_coord=second_interface[a][0]
    y_coord=second_interface[a][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    return r,c 

cn = nx = int(mesh_data["nx"])  #201
rn = ny = int(mesh_data["ny"]) 
print(del_h,nx,ny)

mat = np.full((nx, ny), np.nan, dtype=object)       # mesh for the solver

# numeric 2D mesh

u_mat = np.full((rn, cn), np.nan)   # u_velocity

v_mat = np.full((rn, cn), np.nan)   # v_velocity

p_mat = np.full((rn, cn), np.nan)   # pressure


print(mat)

variable_array = []                                 # variable marker mesh (in use for pressure)
for i in range (0,len(inside_pt),1):
    r,c = cord_transfer_logic(i)
    x = f'x{r}|{c}'       # beware of row and column to x-y coordinate system
    variable_array.append(x)
# print("🐒: ",len(variable_array),variable_array)


variable_array_copy = variable_array.copy()         # all the pressure BC editing is done on it. Make ghost_node value == first_interface

# Appending all the initial conditions to respective mesh nodes u, v and p (for fluid nodes)
for i in range(0,len(inside_pt),1):
    r,c = cord_transfer_logic(i)
    u_mat[r][c] = 0                 # uniform initial velocity (u) condition through out the geometry
    v_mat[r][c] = 0                 # uniform initial velocity (v) condition through out the geometry
    p_mat[r][c] = 0                 # uniform initial pressure condition
    


drich_u = [0,0,0,0,0,0]             # x direction velocity drichilit boundary condition
drich_v = [0,0,0,0,0,0]             # y direction velocity drichilit boundary condition
drich_p = [0,0,0,0,0,0]             # pressure drichilit boundary condition


for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):
        x = ghost_nodes[i][j][0]
        y = ghost_nodes[i][j][1]
        r = int(round((y * conversion_factor),0))
        c = int(round((x * conversion_factor),0))
        u_mat[r][c] = drich_u[i]
        v_mat[r][c] = drich_v[i]
        p_mat[r][c] = drich_p[i]
#=======================================================================================================================================#
#                                                                       END
#=======================================================================================================================================#

#=======================================================================================================================================#
#                                                       SOME EXAMPLES OF BOUNDARY CONDITIONS
#=======================================================================================================================================#
# BCs for some example problems
# ======================================Backward Facing Step=================================================
# drich_u = [0,0,0,"NDN",0,1]                                 # u velocity drichilit boundary condition
# drich_v = [0,0,0,"NDN",0,0]                                 # v velocity drichilit boundary condition
# drich_p = ["NDN","NDN","NDN",0,"NDN","NDN"]                 # p pressure drichilit boundary condition

   
# Ne_BC_u = ["NCN","NCN","NCN",0,"NCN","NCN"]                 # u velocity drichilit boundary condition
# Ne_BC_v = ["NCN","NCN","NCN",0,"NCN","NCN"]                 # v velocity drichilit boundary condition
# Ne_BC_p = [0,0,0,"NCN",0,0]                                 # p velocity drichilit boundary condition
# # ==============================================Channel Flow================================================
# drich_u = [0,"NDN",0,1]                         # u velocity drichilit boundary condition
# drich_v = [0,0,0,0]                             # v velocity drichilit boundary condition
# drich_p = ["NDN",0,"NDN","NDN"]                 # p pressure drichilit boundary condition

   
# Ne_BC_u = ["NCN",0,"NCN","NCN"]                 # u velocity drichilit boundary condition
# Ne_BC_v = ["NCN","NCN","NCN","NCN"]             # v velocity drichilit boundary condition
# Ne_BC_p = [0,"NCN",0,0]                         # p velocity drichilit boundary condition
# #==========================================Lid driven Cavity=======================================
# drich_u = [0,0,1,0]                                 # u velocity drichilit boundary condition
# drich_v = [0,0,0,0]                                 # v velocity drichilit boundary condition
# drich_p = ["NDN","NDN","NDN","NDN"]                 # p pressure drichilit boundary condition

   
# Ne_BC_u = ["NCN","NCN","NCN","NCN"]                 # u velocity drichilit boundary condition
# Ne_BC_v = ["NCN","NCN","NCN","NCN"]                 # v velocity drichilit boundary condition
# Ne_BC_p = [0,0,0,0]                                 # p velocity drichilit boundary condition
#=======================================================================================================================================#
#                                                        SETTING UP BOUNDARY CONDITIONS
# #=======================================================================================================================================#

drich_u = [0,0,1,0]                                 # u velocity drichilit boundary condition
drich_v = [0,0,0,0]                                 # v velocity drichilit boundary condition
drich_p = ["NDN","NDN","NDN","NDN"]                 # p pressure drichilit boundary condition

   
Ne_BC_u = ["NCN","NCN","NCN","NCN"]                 # u velocity drichilit boundary condition
Ne_BC_v = ["NCN","NCN","NCN","NCN"]                 # v velocity drichilit boundary condition
Ne_BC_p = [0,0,0,0]                                 # p velocity drichilit boundary condition

for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):
        x = ghost_nodes[i][j][0]
        y = ghost_nodes[i][j][1]
        r = int(round((y * conversion_factor),0))
        c = int(round((x * conversion_factor),0))
        # print(r,c,"mmmm")
        if (drich_u[i] != "NDN"):
            u_mat[r][c] = drich_u[i]
        if (drich_v[i] != "NDN"):
            v_mat[r][c] = drich_v[i]
        if (drich_p[i] != "NDN"):    
            p_mat[r][c] = drich_p[i]

#=======================================================================================================================================#
#                                                               END
#=======================================================================================================================================#

#=========================================================== Geometry check ============================================================#
lowerlimit = 0
upperlimit = 1
x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(cn))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(rn))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
Z = u_mat  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
# contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
plt.colorbar(contour, ax=ax, label='u Velocity')
# ax.streamplot(X, Y, u_stack[timestep], v_stack[timestep], color= 'k', density=1.5, linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#=======================================================================================================================================#
# target = (0.0787,9.9949)
# import numpy as np
# target = (7.5,2.0)
# target = np.array(target)

# matches = np.where((inside_pt == target).all(axis=1))[0]

# if len(matches) == 0:
#     print("Target not found:", target)
# else:
#     idx = matches[0]
#     print("Target index:", idx)

# target = (0.0787,9.9162)
# import numpy as np

# target = np.array(target)

# matches = np.where((inside_pt == target).all(axis=1))[0]

# if len(matches) == 0:
#     print("Target not found:", target)
# else:
#     idx = matches[0]
#     print("Target index:", idx)

# target = (0.1574,9.9949)
# import numpy as np

# target = np.array(target)

# matches = np.where((inside_pt == target).all(axis=1))[0]

# if len(matches) == 0:
#     print("Target not found:", target)
# else:
#     idx = matches[0]
#     print("Target index:", idx)

# time.sleep(900)

total_time_steps = 500000
del_t = 0.001

del_h = 1.0/(nx-1)
print("△h = ",del_h)

Re = 400        # Set Reynolds number here
etr = 100        # Set etr here (every time iteration)
RESTART = 0   # 1 (restart) 0 (no restart)


u_old = u_mat
v_old = v_mat
p_old = p_mat

# del_h = 0.00787
# time.sleep(800)

sst_main = time.time()
#--------------------------------------------------------------------------------------------------------#
B_vector_sequence = []
#---------------------------------------start of time loop-----------------------------------------------#

u_p = u_mat.copy()           # u_velocity copy mesh
v_p = v_mat.copy()           # v_velocity copy mesh
    
u_copy = u_mat.copy()
v_copy = v_mat.copy() 

if (RESTART == 1):

    start = 321101
    # =====================================
    # USER: Put file path here manually
    # =====================================

    file_path_u = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_u/Time_stack_u_t321100.npz"
    file_path_v = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_v/Time_stack_v_t321100.npz"
    file_path_p = r"D:/numerical computation/geometry meshing/Meshes/Time_stack_p/Time_stack_p_t321100.npz"

    # =====================================
    # Load NPZ file
    # =====================================

    data_u = np.load(file_path_u)
    data_v = np.load(file_path_v)
    data_p = np.load(file_path_p)

    print("Available arrays:", data_u.files)
    print("Available arrays:", data_v.files)
    print("Available arrays:", data_p.files)

    key_u = data_u.files[0]       # only one array expected
    key_v = data_v.files[0]       # only one array expected
    key_p = data_p.files[0]       # only one array expected

    U = data_u[key_u]
    V = data_v[key_v]
    P = data_p[key_p]

    u_old = U
    v_old = V
    p_old = P

else:
    start = 1
    pass
    
print(inside_pt)
print(variable_array)
# time.sleep(700)
#=======================================================================================================================================#
#                                                           TIME LOOP BEGINS
#=======================================================================================================================================#
for t in range(start, total_time_steps, 1):
    
    print("=================================================================================================")
    print("itn = ",t,"/",total_time_steps)

    for i in range(0,len(first_interface),1):
        ip,jp = cord_transfer_logic_l1(i)
        u = u_old[ip][jp]
        v = v_old[ip][jp]

        # u * ∂u/∂x term
        if u >= 0:
            u_du_dx = u * (u_old[ip][jp] - u_old[ip][jp-1]) / del_h
        else:
            u_du_dx = u * (u_old[ip][jp+1] - u_old[ip][jp]) / del_h

        # v * ∂u/∂y term
        if v >= 0:
            v_du_dy = v * (u_old[ip][jp] - u_old[ip-1][jp]) / del_h
        else:
            v_du_dy = v * (u_old[ip+1][jp] - u_old[ip][jp]) / del_h

        Hu_conv = u_du_dx + v_du_dy

        # For v convection term (Hv_conv)
        # u * ∂v/∂x term
        if u >= 0:
            u_dv_dx = u * (v_old[ip][jp] - v_old[ip][jp-1]) / del_h
        else:
            u_dv_dx = u * (v_old[ip][jp+1] - v_old[ip][jp]) / del_h

        # v * ∂v/∂y term
        if v >= 0:
            v_dv_dy = v * (v_old[ip][jp] - v_old[ip-1][jp]) / del_h
        else:
            v_dv_dy = v * (v_old[ip+1][jp] - v_old[ip][jp]) / del_h

        Hv_conv = u_dv_dx + v_dv_dy
        # print("I: ",u_old[ip-1][jp])
        #-----------------------------------------------Diffusive flux (Second order central difference)-------------------------------------------------------------------------------------------------------------------#
        Hu_diffusion = (1.0/Re)*((u_old[ip][jp+1] + u_old[ip][jp-1] + u_old[ip+1][jp] + u_old[ip-1][jp] - 4*u_old[ip][jp])/(del_h**2))  
        Hv_diffusion = (1.0/Re)*((v_old[ip][jp+1] + v_old[ip][jp-1] + v_old[ip+1][jp] + v_old[ip-1][jp] - 4*v_old[ip][jp])/(del_h**2)) 
        #-----------------------------------------------------------Final flux (Up)-------------------------------------------------------------------------------------------------------------------#
        Up =  (u_old[ip][jp]) + del_t*(-1*Hu_conv + Hu_diffusion)        # Up = Un + △t*(-Cu + Du)
        u_p[ip][jp] = Up
        Vp = (v_old[ip][jp]) + del_t*(-1*Hv_conv + Hv_diffusion)         # Vp = Vn + △t*(-Cv + Dv)
        v_p[ip][jp] = Vp         

    for i in range(0,len(second_interface),1):
        ip,jp = cord_transfer_logic_l2(i)
        u = u_old[ip][jp]
        v = v_old[ip][jp]

        # u * ∂u/∂x term with 2nd order upwinding
        if u >= 0:
            # Backward difference (2nd order) for u >= 0
            u_du_dx = u * (3*u_old[ip][jp] - 4*u_old[ip][jp-1] + u_old[ip][jp-2]) / (2*del_h)
        else:
            # Forward difference (2nd order) for u < 0
            u_du_dx = u * (-3*u_old[ip][jp] + 4*u_old[ip][jp+1] - u_old[ip][jp+2]) / (2*del_h)

        # v * ∂u/∂y term with 2nd order upwinding
        if v >= 0:
            # Backward difference (2nd order) for v >= 0
            v_du_dy = v * (3*u_old[ip][jp] - 4*u_old[ip-1][jp] + u_old[ip-2][jp]) / (2*del_h)
        else:
            # Forward difference (2nd order) for v < 0
            v_du_dy = v * (-3*u_old[ip][jp] + 4*u_old[ip+1][jp] - u_old[ip+2][jp]) / (2*del_h)

        Hu_conv = u_du_dx + v_du_dy

        # For v convection term (Hv_conv) with 2nd order upwinding
        # u * ∂v/∂x term
        if u >= 0:
            # Backward difference (2nd order) for u >= 0
            u_dv_dx = u * (3*v_old[ip][jp] - 4*v_old[ip][jp-1] + v_old[ip][jp-2]) / (2*del_h)
        else:
            # Forward difference (2nd order) for u < 0
            u_dv_dx = u * (-3*v_old[ip][jp] + 4*v_old[ip][jp+1] - v_old[ip][jp+2]) / (2*del_h)

        # v * ∂v/∂y term
        if v >= 0:
            # Backward difference (2nd order) for v >= 0
            v_dv_dy = v * (3*v_old[ip][jp] - 4*v_old[ip-1][jp] + v_old[ip-2][jp]) / (2*del_h)
        else:
            # Forward difference (2nd order) for v < 0
            v_dv_dy = v * (-3*v_old[ip][jp] + 4*v_old[ip+1][jp] - v_old[ip+2][jp]) / (2*del_h)

        Hv_conv = u_dv_dx + v_dv_dy
        # print("I: ",u_old[ip-1][jp])
        #-----------------------------------------------Diffusive flux (Second order central difference)-------------------------------------------------------------------------------------------------------------------#
        Hu_diffusion = (1.0/Re)*((u_old[ip][jp+1] + u_old[ip][jp-1] + u_old[ip+1][jp] + u_old[ip-1][jp] - 4*u_old[ip][jp])/(del_h**2))  
        Hv_diffusion = (1.0/Re)*((v_old[ip][jp+1] + v_old[ip][jp-1] + v_old[ip+1][jp] + v_old[ip-1][jp] - 4*v_old[ip][jp])/(del_h**2)) 
        #-----------------------------------------------------------Final flux (Up)-------------------------------------------------------------------------------------------------------------------#
        Up =  (u_old[ip][jp]) + del_t*(-1*Hu_conv + Hu_diffusion)        # Up = Un + △t*(-Cu + Du)
        u_p[ip][jp] = Up
        Vp = (v_old[ip][jp]) + del_t*(-1*Hv_conv + Hv_diffusion)         # Vp = Vn + △t*(-Cv + Dv)
        v_p[ip][jp] = Vp         
    
    for i in range(0,len(sorted_first_interface),1):
        for j in range(0,len(sorted_first_interface[i]),1):
            if (Ne_BC_u[i] != "NCN"):
                xf = sorted_first_interface[i][j][0]
                yf = sorted_first_interface[i][j][1]
                rf = int(round((yf * conversion_factor),0))
                cf = int(round((xf * conversion_factor),0))
                xgn = ghost_nodes[i][j][0]
                ygn = ghost_nodes[i][j][1]
                rgn = int(round((ygn * conversion_factor),0))
                cgn = int(round((xgn * conversion_factor),0))
                u_p[rgn][cgn] = u_p[rf][cf]
            if (Ne_BC_v[i] != "NCN"):
                xf = sorted_first_interface[i][j][0]
                yf = sorted_first_interface[i][j][1]
                rf = int(round((yf * conversion_factor),0))
                cf = int(round((xf * conversion_factor),0))
                xgn = ghost_nodes[i][j][0]
                ygn = ghost_nodes[i][j][1]
                rgn = int(round((ygn * conversion_factor),0))
                cgn = int(round((xgn * conversion_factor),0))
                v_p[rgn][cgn] = v_p[rf][cf]


    for i in range(0,len(ghost_nodes),1):
        for j in range(0,len(ghost_nodes[i]),1):
            x = ghost_nodes[i][j][0]
            y = ghost_nodes[i][j][1]
            r = int(round((y * conversion_factor),0))
            c = int(round((x * conversion_factor),0))
            if(drich_u[i] != "NDN"): 
                u_p[r][c] = drich_u[i]
            if (drich_v[i] != "NDN"):
                v_p[r][c] = drich_v[i]
            else:
                pass
#========================================================================================================================================#
#                                                               Building [A][p'] = [B]
#========================================================================================================================================#
    # for moving and deforming bodies
    # inside_pt = time_based_inside_pt[t]     # here inside points which are going to be function of time are stored in time_based_inside_pt
    # ghost_node = time_based_ghost_node[t]   # here ghost points which are going to be function of time are stored in time_based_ghost_node
    if(t == start):
        sat = time.time()
        # Building co-efficient matrix A here
        A = np.zeros((len(variable_array),len(variable_array)))
        B = []
        for i in range(0,len(variable_array),1):
            x_coord=inside_pt[i][0]
            y_coord=inside_pt[i][1]
            # At the point (x_coord, y_coord)
            # print("<M>",x_coord,y_coord)
            row = int(round((y_coord * conversion_factor),0))
            col = int(round((x_coord * conversion_factor),0)) 
            # print(">M<",row,col)
            io=row
            jo=col 
            # Find the indices of the neighboring points
            east = col+1
            west = col-1
            south = row-1
            north = row+1  
                    
            # Neighbor handling with safe check
            key_east = f'x{row}|{east}'
            key_west = f'x{row}|{west}'
            key_south = f'x{south}|{col}'
            key_north = f'x{north}|{col}'
            # print(key_east)
            # print(key_north)
            # print(key_south)
            # print(key_west)

            a = [-4]
            b_e = []
            b_vector_data = []
            if key_east in variable_array:
                east_m = variable_array.index(key_east)
                A[i][east_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t =  round((east/conversion_factor),4)
                    y_t = round((row/conversion_factor),4)
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC_p[ne_pos] != "NCN"):
                            b_e.append(Ne_BC_p[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            a.append(1)
                            if (x_coord == 2 and y_coord==2):
                                print("HI-1!!!!",Ne_BC_p[ne_pos])
                        else:
                            b_e.append(0)
                            b_vector_data.append(0)
                            pass
                        break
                    else:
                        pass
               
            if key_west in variable_array:
                west_m = variable_array.index(key_west)
                A[i][west_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = round((west / conversion_factor),4)
                    y_t = round((row/ conversion_factor),4)
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC_p[ne_pos] != "NCN"):
                            b_e.append(Ne_BC_p[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            a.append(1)
                            if (x_coord == 2 and y_coord == 2):
                                print("HI-2!!!!",Ne_BC_p[ne_pos])
                        else:
                            b_e.append(0)
                            b_vector_data.append(0)
                            pass
                        break
                    else:
                        pass
                      
            if key_south in variable_array:
                south_m = variable_array.index(key_south)
                A[i][south_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = round((col / conversion_factor),4)
                    y_t = round((south/ conversion_factor),4)
                    target = (x_t,y_t)
                    # need to convert target back into x-y coordinate
                    # print("down",target,key_south,key_west)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        # print(";;;;")
                        ne_pos = ghost_nodes_list.index(current_sub_gn)      # this line tells which edge we are dealng with
                        if (Ne_BC_p[ne_pos] != "NCN"):        # this implies a neuman condition exist 
                            b_e.append(Ne_BC_p[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            a.append(1)                         
                            if (x_coord == 2 and y_coord==2):
                                print("HI-3!!!!",Ne_BC_p[ne_pos])
                                print("NNNN: ",ne_pos)
                        else:                                  # this implies a drichilet condition exist (p' = p[n+1]-p[n] = 0)
                            b_e.append(0)
                            b_vector_data.append(0)
                            pass
                        break
                    else:
                        pass
        
            if key_north in variable_array:
                north_m = variable_array.index(key_north)
                A[i][north_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = round((col / conversion_factor),4)
                    y_t = round((north/ conversion_factor),4)
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC_p[ne_pos] != "NCN"):
                            b_e.append(Ne_BC_p[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC_p[ne_pos])
                            if (x_coord == 2 and y_coord== 2):
                                print("HI-4!!!!",Ne_BC_p[ne_pos])
                            a.append(1)                     # appending 1 in "a"
                        else:
                            b_e.append(0)
                            b_vector_data.append(0)
                            pass
                        break
                    else:
                        pass
      
            diag_a = np.sum(a)
            A[i][i] = diag_a
            Avg_b1 = (u_p[io][jo+1] - u_p[io][jo-1])/(2)
            Avg_b2 = (v_p[io+1][jo] - v_p[io-1][jo])/(2) 
            zeta = del_t/del_h
            const = (del_h**2/(del_t*del_h)) * (  (Avg_b1  +  2*zeta*p_old[io][jo] - zeta*p_old[io][jo-1] - zeta*p_old[io][jo+1])
                                                + (Avg_b2 + 2*zeta*p_old[io][jo] - zeta*p_old[io-1][jo] - zeta*p_old[io+1][jo]) )            
            
            b_e.append(const)
            b_final = np.sum(b_e)
            B.append(b_final)
            B_vector_sequence.append(b_vector_data)
        

        B_np = np.array(B, dtype=np.float64)
        A_np = np.array(A, dtype=np.float32)
        eat = time.time()
        print("A_time: ",eat-sat)
      
    
        print("--------pop-------")
        print("A matrix")
        print(A_np)
        # print(A_np[75]) 
        # print(A_np[113]) 
        # time.sleep(500) 
        # A_np[16002,:] = 0
        # A_np[16002][16002] = 1
        # A_np[16003][16002] = 0
        # A_np[15875][16002]= 0
        # B_np[16002] = 0
        # time.sleep(200)

    if(t > start):
        sbt = time.time()
        # print("?><:?>:",len(inside_pt))
        # print("?><:?>:",len(B_vector_sequence))
        # time.sleep(2)
        B_sub = []
        for i in range(0,len(inside_pt),1):
            row,col = cord_transfer_logic(i)
            # print(">M<",row,col)
            io=row
            jo=col 
            Avg_b1 = (u_p[io][jo+1] - u_p[io][jo-1])/(2)  # (Up[j+1] - Up[j-1])/2
            Avg_b2 = (v_p[io+1][jo] - v_p[io-1][jo])/(2)  # (Vp[i-1] - Vp[i-1])/2
            zeta = del_t/del_h
            const = (del_h**2/(del_t*del_h)) * (  (Avg_b1  +  2*zeta*p_old[io][jo] - zeta*p_old[io][jo-1] - zeta*p_old[io][jo+1])
                                                + (Avg_b2 + 2*zeta*p_old[io][jo] - zeta*p_old[io-1][jo] - zeta*p_old[io+1][jo]) )

            b = B_vector_sequence[i]
            b_sum = const + np.sum(b)
            B_sub.append(b_sum)
        B_np = np.array(B_sub, dtype=np.float64)
        B_gpu = cp.asarray(B_np)        # move B_np to GPU

        ebt = time.time()
        # print("B_time::",ebt-sbt)
    A_np[6162,:] = 0
    A_np[6162][6162] = 1
    A_np[6162][6163] = 0
    A_np[6083][6162]= 0
    B_np[6162] = 0
    
    st = time.time()
    #---------------------------------------------------------------------------------------------------------------#
    #                                                 Gauss-Seidel Method
    #---------------------------------------------------------------------------------------------------------------#
    # print("Time taken for A & B matrix generation = ",ee-ss)
    # check for SDD
    
    if (t==start):
        # gauss siedel method
        s = np.tril(A_np)               # lower triangular matrix (contains the diagonal element)
        s_inv = np.linalg.inv(s)
        D = np.diag(np.diag(A_np))
        T = np.triu(A_np)-D             # STRICTLY upper triangular matrix (contains only element above the diagonal)
        # del A_np

        # ============================================================
        # MOVE TO GPU (only once!)
        # ============================================================
        print(f"Moving matrices to GPU ({device})...")
        s_inv_gpu = torch.from_numpy(s_inv).float().to(device)
        T_gpu = torch.from_numpy(T).float().to(device)
        # print(f"✓ Matrices on GPU, size: {s_inv.shape}")
        
        # Check memory
        if device == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1e9
            print(f"GPU memory used: {memory_used:.2f} GB")
    
    # ============================================================
    # EVERY TIME STEP: Convert RHS to GPU
    # ============================================================
    B_gpu = torch.from_numpy(B_np).float().to(device)
    x_current_gpu = torch.zeros(len(variable_array), device=device)
    
    # ============================================================
    # GAUSS-SEIDEL ON GPU
    # ============================================================
    tol = 1e-3
    
    # Synchronize before timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    stgs = time.time()
    
    for l in range(500):
        # if l % 50 == 0:  # Print every 50 iterations to save time
            # print(f"\rTime step {t}/{total_time_steps}, GS iteration: {l}/500", end="", flush=True)
        
        # Store old value
        x_old_gpu = x_current_gpu.clone()
        
        # Gauss-Seidel formula (all on GPU!)
        x_current_gpu = s_inv_gpu @ (B_gpu - (T_gpu @ x_old_gpu))
        
        # Calculate error
        error = torch.norm(x_current_gpu - x_old_gpu, p=float('inf')).item()
        
        # Check convergence
        if error < tol:
            print("Convergence achieved in iteration = ",l)
            break
    
    # Synchronize after computation
    if device == 'cuda':
        torch.cuda.synchronize()
    
    gs_time = time.time() - stgs
    # print(f" | GS time: {gs_time:.4f}s")
    
    # ============================================================
    # MOVE SOLUTION BACK TO CPU for rest of calculations
    # ============================================================
    solution_vector = x_current_gpu.cpu().numpy().astype(np.float64)
    
    # Continue with your pressure update, velocity correction, etc.
    # ... rest of your code using solution_vector (now on CPU) ...
    #--------------------------------------------------------------------------------------------------------------------------------#
    #                                                       END
    #--------------------------------------------------------------------------------------------------------------------------------#
    # solution_vector, info = spla.cg(A_csr_gpu, B_gpu, tol = 1e-3, maxiter = 500)   # Conjugate gradient
    print(Fore.YELLOW + "Final-Solution" + Style.RESET_ALL)
    print(solution_vector)
    print("lop: ",solution_vector[6162])
    # print(solution_vector[75])
    # print(A_np[113])
    # time.sleep(800)
        
    p_prime = p_mat.copy()                   
    for i in range(0,len(solution_vector),1):
        r,c = cord_transfer_logic(i)
        p_prime[r][c] = solution_vector[i]         # uniform initial boundary condition through out the geometry
    
 
    for i in range(0,len(sorted_first_interface),1):
        for j in range(0,len(sorted_first_interface[i]),1):
            if (Ne_BC_p[i] != "NCN"):
                xf = sorted_first_interface[i][j][0]
                yf = sorted_first_interface[i][j][1]
                rf = int(round((yf * conversion_factor),0))
                cf = int(round((xf * conversion_factor),0))
                xgn = ghost_nodes[i][j][0]
                ygn = ghost_nodes[i][j][1]
                rgn = int(round((ygn * conversion_factor),0))
                cgn = int(round((xgn * conversion_factor),0))
                p_prime[rgn][cgn] = p_prime[rf][cf]

    for i in range(0,len(ghost_nodes),1):
        for j in range(0,len(ghost_nodes[i]),1):
            x = ghost_nodes[i][j][0]
            y = ghost_nodes[i][j][1]
            r = int(round((y * conversion_factor),0))
            c = int(round((x * conversion_factor),0))
            if(drich_p[i] != "NDN"): 
                p_prime[r][c] = drich_p[i]
            else:
                pass
            
    for i in range(0,len(ghost_nodes),1):
        for j in range(0,len(ghost_nodes[i]),1):
            x = ghost_nodes[i][j][0]
            y = ghost_nodes[i][j][1]
            r = int(round((y * conversion_factor),0))
            c = int(round((x * conversion_factor),0))
            if(drich_u[i] != "NDN"): 
                u_copy[r][c] = drich_u[i]
            if (drich_v[i] != "NDN"):
                v_copy[r][c] = drich_v[i]
            else:
                pass

    p_new = p_prime + p_old        #corrected pressure copy mesh     # p(n+1) = p' + p(n)
    # p_stack.append(p_new)
    p_old = p_new

    for i in range(0,len(first_interface),1):
        ib,jb = cord_transfer_logic_l1(i)
        #========================================For u convection term (Hu_conv)==================================
        u = u_old[ib][jb]
        v = v_old[ib][jb]

        # u * ∂u/∂x term
        if u >= 0:
            u_du_dx = u * (u_old[ib][jb] - u_old[ib][jb-1]) / del_h
        else:
            u_du_dx = u * (u_old[ib][jb+1] - u_old[ib][jb]) / del_h

        # v * ∂u/∂y term
        if v >= 0:
            v_du_dy = v * (u_old[ib][jb] - u_old[ib-1][jb]) / del_h
        else:
            v_du_dy = v * (u_old[ib+1][jb] - u_old[ib][jb]) / del_h

        Hu_conv = u_du_dx + v_du_dy

        # For v convection term (Hv_conv)
        # u * ∂v/∂x term
        if u >= 0:
            u_dv_dx = u * (v_old[ib][jb] - v_old[ib][jb-1]) / del_h
        else:
            u_dv_dx = u * (v_old[ib][jb+1] - v_old[ib][jb]) / del_h

        # v * ∂v/∂y term
        if v >= 0:
            v_dv_dy = v * (v_old[ib][jb] - v_old[ib-1][jb]) / del_h
        else:
            v_dv_dy = v * (v_old[ib+1][jb] - v_old[ib][jb]) / del_h

        Hv_conv = u_dv_dx + v_dv_dy
        # print("?/>",queta_c," ",qveta_c)
        #-----------------------------------------------Diffusive flux (Second order central difference)-------------------------------------------------------------------------------------------------------------------#
        Hu_diffusion = (1/Re)*((u_old[ib][jb+1] + u_old[ib][jb-1] + u_old[ib+1][jb] + u_old[ib-1][jb] - 4*u_old[ib][jb])/(del_h**2))  
        Hv_diffusion = (1/Re)*((v_old[ib][jb+1] + v_old[ib][jb-1] + v_old[ib+1][jb] + v_old[ib-1][jb] - 4*v_old[ib][jb])/(del_h**2)) 
        #---------------------------------------------------------#
        u_copy[ib][jb] = u_old[ib][jb] + del_t*(-1*Hu_conv + Hu_diffusion - ((p_new[ib][jb+1] - p_new[ib][jb-1])/(2*del_h)))
        v_copy[ib][jb] = v_old[ib][jb] + del_t*(-1*Hv_conv + Hv_diffusion - ((p_new[ib+1][jb] - p_new[ib-1][jb])/(2*del_h)))
        # if (jb==15 and t==179):
        #     v_check.append(u_copy[ib][jb])

    for i in range(0,len(second_interface),1):
        ib,jb = cord_transfer_logic_l2(i)
        u = u_old[ib][jb]
        v = v_old[ib][jb]

        # u * ∂u/∂x term with 2nd order upwinding
        if u >= 0:
            # Backward difference (2nd order) for u >= 0
            u_du_dx = u * (3*u_old[ib][jb] - 4*u_old[ib][jb-1] + u_old[ib][jb-2]) / (2*del_h)
        else:
            # Forward difference (2nd order) for u < 0
            u_du_dx = u * (-3*u_old[ib][jb] + 4*u_old[ib][jb+1] - u_old[ib][jb+2]) / (2*del_h)

        # v * ∂u/∂y term with 2nd order upwinding
        if v >= 0:
            # Backward difference (2nd order) for v >= 0
            v_du_dy = v * (3*u_old[ib][jb] - 4*u_old[ib-1][jb] + u_old[ib-2][jb]) / (2*del_h)
        else:
            # Forward difference (2nd order) for v < 0
            # print(second_interface[i],ib,jb, ib/conversion_factor,jb/conversion_factor)
            v_du_dy = v * (-3*u_old[ib][jb] + 4*u_old[ib+1][jb] - u_old[ib+2][jb]) / (2*del_h)


        Hu_conv = u_du_dx + v_du_dy

        # For v convection term (Hv_conv) with 2nd order upwinding
        # u * ∂v/∂x term
        if u >= 0:
            # Backward difference (2nd order) for u >= 0
            u_dv_dx = u * (3*v_old[ib][jb] - 4*v_old[ib][jb-1] + v_old[ib][jb-2]) / (2*del_h)
        else:
            # Forward difference (2nd order) for u < 0
            u_dv_dx = u * (-3*v_old[ib][jb] + 4*v_old[ib][jb+1] - v_old[ib][jb+2]) / (2*del_h)

        # v * ∂v/∂y term
        if v >= 0:
            # Backward difference (2nd order) for v >= 0
            v_dv_dy = v * (3*v_old[ib][jb] - 4*v_old[ib-1][jb] + v_old[ib-2][jb]) / (2*del_h)
        else:
            # Forward difference (2nd order) for v < 0
            v_dv_dy = v * (-3*v_old[ib][jb] + 4*v_old[ib+1][jb] - v_old[ib+2][jb]) / (2*del_h)

        Hv_conv = u_dv_dx + v_dv_dy
        # print("I: ",u_old[ip-1][jp])
        #-----------------------------------------------Diffusive flux (Second order central difference)-------------------------------------------------------------------------------------------------------------------#
        Hu_diffusion = (1.0/Re)*((u_old[ib][jb+1] + u_old[ib][jb-1] + u_old[ib+1][jb] + u_old[ib-1][jb] - 4*u_old[ib][jb])/(del_h**2))  
        Hv_diffusion = (1.0/Re)*((v_old[ib][jb+1] + v_old[ib][jb-1] + v_old[ib+1][jb] + v_old[ib-1][jb] - 4*v_old[ib][jb])/(del_h**2)) 
        #-----------------------------------------------------------
        u_copy[ib][jb] = u_old[ib][jb] + del_t*(-1*Hu_conv + Hu_diffusion - ((p_new[ib][jb+1] - p_new[ib][jb-1])/(2*del_h)))
        v_copy[ib][jb] = v_old[ib][jb] + del_t*(-1*Hv_conv + Hv_diffusion - ((p_new[ib+1][jb] - p_new[ib-1][jb])/(2*del_h)))


    for i in range(0,len(sorted_first_interface),1):
        for j in range(0,len(sorted_first_interface[i]),1):
            if (Ne_BC_u[i] != "NCN"):
                xf = sorted_first_interface[i][j][0]
                yf = sorted_first_interface[i][j][1]
                rf = int(round((yf * conversion_factor),0))
                cf = int(round((xf * conversion_factor),0))
                xgn = ghost_nodes[i][j][0]
                ygn = ghost_nodes[i][j][1]
                rgn = int(round((ygn * conversion_factor),0))
                cgn = int(round((xgn * conversion_factor),0))
                u_copy[rgn][cgn] = u_copy[rf][cf]
            if (Ne_BC_v[i] != "NCN"):
                xf = sorted_first_interface[i][j][0]
                yf = sorted_first_interface[i][j][1]
                rf = int(round((yf * conversion_factor),0))
                cf = int(round((xf * conversion_factor),0))
                xgn = ghost_nodes[i][j][0]
                ygn = ghost_nodes[i][j][1]
                rgn = int(round((ygn * conversion_factor),0))
                cgn = int(round((xgn * conversion_factor),0))
                v_copy[rgn][cgn] = v_copy[rf][cf]

    for i in range(0,len(ghost_nodes),1):
        for j in range(0,len(ghost_nodes[i]),1):
            x = ghost_nodes[i][j][0]
            y = ghost_nodes[i][j][1]
            r = int(round((y * conversion_factor),0))
            c = int(round((x * conversion_factor),0))
            if(drich_u[i] != "NDN"): 
                u_copy[r][c] = drich_u[i]
            if (drich_v[i] != "NDN"):
                v_copy[r][c] = drich_v[i]
            else:
                pass
        
    # updating u_old and v_old
    u_old = u_copy
    v_old = v_copy
    # print(etr)
    #===========================SAVING DATA===============================#
    if t % etr < 1e-3:

        base_dir = r"D:/numerical computation/geometry meshing/Meshes"

        # ----------------------------
        # Create folders once (safe)
        # ----------------------------
        u_dir = os.path.join(base_dir, "Time_stack_u")
        v_dir = os.path.join(base_dir, "Time_stack_v")
        p_dir = os.path.join(base_dir, "Time_stack_p")

        os.makedirs(u_dir, exist_ok=True)
        os.makedirs(v_dir, exist_ok=True)
        os.makedirs(p_dir, exist_ok=True)

        print("@",t)
        tag = f"{t:06d}"   # zero padded timestep
        print("@",tag)

        # ----------------------------
        # Force numeric arrays (important)
        # ----------------------------
        u_clean = np.asarray(u_old, dtype=np.float32)
        v_clean = np.asarray(v_old, dtype=np.float32)
        p_clean = np.asarray(p_old, dtype=np.float32)

        # ----------------------------
        # Save with UNIQUE filenames
        # ----------------------------
        u_path = os.path.join(u_dir, f"Time_stack_u_t{tag}.npz")
        v_path = os.path.join(v_dir, f"Time_stack_v_t{tag}.npz")
        p_path = os.path.join(p_dir, f"Time_stack_p_t{tag}.npz")

        np.savez(u_path, u=u_clean)
        np.savez(v_path, v=v_clean)
        np.savez(p_path, p=p_clean)

        print(f"✅ Saved timestep {t}")



    print("===================================================================================================")
    # time.sleep(900)
eet_main = time.time()
print("")
print("Net calculation time = ",eet_main-sst_main)


#=======================================================================================================================================#
#                                                               POST-PROCESSING
#=======================================================================================================================================#
lowerlimit = 0
upperlimit = 1

x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(cn))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(rn))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
timestep = -1
Z = u_old  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
# contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
plt.colorbar(contour, ax=ax, label='u Velocity')
ax.streamplot(X, Y, u_old, v_old, color= 'k', density = 1.5, linewidth=1)
plt.title("Lid Driven Cavity: Velocity Streamlines")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(cn))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(rn))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
timestep = -1
Z = v_old  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
# plt.colorbar(contour, ax=ax, label='u Velocity')
# ax.streamplot(X, Y, u_stack[timestep], v_stack[timestep], color= 'k', density=1.5, linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.show()



x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(cn))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(rn))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
timestep = -1
Z = p_old  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, 20, cmap='coolwarm')   # u velocity plot
contour = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.8)
# plt.colorbar(contour, ax=ax, label='u Velocity')
# ax.streamplot(X, Y, u_stack[timestep], v_stack[timestep], color= 'k', density=1.5, linewidth=1)
plt.title("Lid Driven Cavity: Pressure plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


