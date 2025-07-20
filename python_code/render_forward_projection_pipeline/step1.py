# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial.transform import Rotation

# global cube_translation, camera_translation

# # Define cube vertices centered at (0,0,0)
# def define_cube(L):
#     L_half = L / 2
#     vertices = np.array([
#         [L_half, L_half, L_half],
#         [L_half, L_half, -L_half],
#         [L_half, -L_half, L_half],
#         [L_half, -L_half, -L_half],
#         [-L_half, L_half, L_half],
#         [-L_half, L_half, -L_half],
#         [-L_half, -L_half, L_half],
#         [-L_half, -L_half, -L_half]
#     ])
#     # Define edges by connecting vertex indices
#     edges = [
#         (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
#     ]
#     return vertices, edges

# # Define camera frustum vertices (vertex at origin, base at z=H)
# def define_frustum(w, h, H):
#     w_half, h_half = w / 2, h / 2
#     # Frustum vertices: apex at (0,0,0), base at z=H
#     vertices = np.array([
#         [0, 0, 0],           # Apex (camera position)
#         [w_half, h_half, H],   # Top-right
#         [w_half, -h_half, H],  # Bottom-right
#         [-w_half, -h_half, H], # Bottom-left
#         [-w_half, h_half, H]   # Top-left
#     ])
#     # Define edges: apex to base vertices, and base rectangle
#     edges = [
#         (0,1), (0,2), (0,3), (0,4),  # Apex to base
#         (1,2), (2,3), (3,4), (4,1)   # Base rectangle
#     ]
#     return vertices, edges

# # Create rotation matrix from Euler angles (yaw, pitch, roll) in degrees
# def get_rotation_matrix(yaw, pitch, roll):
#     return Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()

# # Apply transformation (rotation + translation) to vertices
# def transform_vertices(vertices, rotation_matrix, translation):
#     # Apply rotation and translation
#     transformed = (rotation_matrix @ vertices.T).T + translation
#     return transformed

# # Plot the 3D scene
# def plot_scene(cube_vertices, cube_edges, frustum_vertices, frustum_edges):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot cube
#     for edge in cube_edges:
#         v0, v1 = cube_vertices[edge[0]], cube_vertices[edge[1]]
#         ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 'b-', label='Cube' if edge == cube_edges[0] else "")

#     # Plot frustum
#     for edge in frustum_edges:
#         v0, v1 = frustum_vertices[edge[0]], frustum_vertices[edge[1]]
#         ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 'r-', label='Frustum' if edge == frustum_edges[0] else "")

#     # Plot image plane as a rectangular grid with gaps
#     w_half, h_half = w / 2, h / 2
#     plane_vertices = frustum_vertices[1:5]  # Base vertices
#     n_subdivisions = 10  # Number of subdivisions for grid
#     u = np.linspace(0, 1, n_subdivisions)
#     v = np.linspace(0, 1, n_subdivisions)
    
#     # Interpolate to create a dense grid of points
#     X = (1-u[:, None]) * (1-v[None, :]) * plane_vertices[0, 0] + u[:, None] * (1-v[None, :]) * plane_vertices[1, 0] + \
#         (1-u[:, None]) * v[None, :] * plane_vertices[3, 0] + u[:, None] * v[None, :] * plane_vertices[2, 0]
#     Y = (1-u[:, None]) * (1-v[None, :]) * plane_vertices[0, 1] + u[:, None] * (1-v[None, :]) * plane_vertices[1, 1] + \
#         (1-u[:, None]) * v[None, :] * plane_vertices[3, 1] + u[:, None] * v[None, :] * plane_vertices[2, 1]
#     Z = (1-u[:, None]) * (1-v[None, :]) * plane_vertices[0, 2] + u[:, None] * (1-v[None, :]) * plane_vertices[1, 2] + \
#         (1-u[:, None]) * v[None, :] * plane_vertices[3, 2] + u[:, None] * v[None, :] * plane_vertices[2, 2]
    
#     # Plot grid lines (horizontal and vertical)
#     for i in range(n_subdivisions):
#         # Horizontal lines (constant v)
#         ax.plot(X[:, i], Y[:, i], Z[:, i], 'g-', linewidth=0.5, label='Image Plane' if i == 0 else "")
#         # Vertical lines (constant u)
#         ax.plot(X[i, :], Y[i, :], Z[i, :], 'g-', linewidth=0.5)

#     # Set labels and title
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('3D Scene: Cube and Camera Frustum')

#     # Equal aspect ratio
#     max_range = np.ptp(np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]), axis=1).max() / 2
#     mid_x = np.mean(ax.get_xlim3d())
#     mid_y = np.mean(ax.get_ylim3d())
#     mid_z = np.mean(ax.get_zlim3d())
#     ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim3d(mid_z - max_range, mid_z + max_range)

#     # ax.scatter(0, 0, 0+0.05, color='black', s=50)
#     # ax.text(0, 0, 0 - 0.5, 'Origin', color='black', fontsize=10)

#     # ax.scatter(camera_translation[0], camera_translation[1], camera_translation[2], color='red', s=50)
#     # ax.text(camera_translation[0] - 0.5, camera_translation[1] + 0.5, camera_translation[2] - 0.5, 'Camera center', color='red', fontsize=10)

#     # ax.text(cube_translation[0], cube_translation[1], cube_translation[2] - 1, 'Cube', color='blue', fontsize=10)

#     ax.set_axis_off()

#     x_range = np.linspace(-10, 10, 19)
#     y_range = np.linspace(-10, 10, 19)
#     X, Y = np.meshgrid(x_range, y_range)
#     Z = np.zeros_like(X)  # z=0 for XY-plane
#     ax.plot_surface(X, Y, Z, color='black', alpha=0.1, label='XY Plane')
#     for i in range(19):
#         ax.plot([x_range[0], x_range[-1]], [y_range[i], y_range[i]], [0, 0], 'black', linewidth=0.5)
#         ax.plot([x_range[i], x_range[i]], [y_range[0], y_range[-1]], [0, 0], 'black', linewidth=0.5)

#     plt.show()

# # Parameters (modify these to tune the scene)
# # Cube parameters
# L = 1  # Cube side length
# # cube_translation = np.array([2, 2, 2])  # [Tx, Ty, Tz]
# cube_translation = np.array([4, 2, 1.5])  # [Tx, Ty, Tz]
# cube_rotation = [30, 60, 90]  # [yaw, pitch, roll] in degrees

# # Camera frustum parameters
# w, h, H = 2, 1.5, 1.5  # Width, height, distance of image plane
# # camera_translation = np.array([-2, 0, 0.5])  # [Cx, Cy, Cz]
# camera_translation = np.array([0, 0, 0])  # [Cx, Cy, Cz]
# camera_rotation = [-34, 56, -51]  # [yaw, pitch, roll] in degrees
# # camera_rotation = [90, 0, 0]  # [yaw, pitch, roll] in degrees

# # Main execution
# if __name__ == "__main__":
#     # Define cube and frustum
#     cube_vertices, cube_edges = define_cube(L)
#     frustum_vertices, frustum_edges = define_frustum(w, h, H)

#     # Get rotation matrices
#     cube_rot_matrix = get_rotation_matrix(*cube_rotation)
#     camera_rot_matrix = get_rotation_matrix(*camera_rotation)

#     # Transform cube and frustum
#     cube_vertices_transformed = transform_vertices(cube_vertices, cube_rot_matrix, cube_translation)
#     frustum_vertices_transformed = transform_vertices(frustum_vertices, camera_rot_matrix, camera_translation)

#     # Plot the scene
#     plot_scene(cube_vertices_transformed, cube_edges, frustum_vertices_transformed, frustum_edges)


# ################################
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # from scipy.spatial.transform import Rotation
# # from matplotlib.widgets import Slider, Button
# # import matplotlib

# # # Ensure an interactive backend
# # matplotlib.use('TkAgg')

# # # Define cube vertices centered at (0,0,0)
# # def define_cube(L):
# #     L_half = L / 2
# #     vertices = np.array([
# #         [L_half, L_half, L_half],
# #         [L_half, L_half, -L_half],
# #         [L_half, -L_half, L_half],
# #         [L_half, -L_half, -L_half],
# #         [-L_half, L_half, L_half],
# #         [-L_half, L_half, -L_half],
# #         [-L_half, -L_half, L_half],
# #         [-L_half, -L_half, -L_half]
# #     ])
# #     edges = [
# #         (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
# #     ]
# #     return vertices, edges

# # # Define camera frustum vertices (vertex at origin, base at z=H)
# # def define_frustum(w, h, H):
# #     w_half, h_half = w / 2, h / 2
# #     vertices = np.array([
# #         [0, 0, 0],           # Apex (camera position)
# #         [w_half, h_half, H],   # Top-right
# #         [w_half, -h_half, H],  # Bottom-right
# #         [-w_half, -h_half, H], # Bottom-left
# #         [-w_half, h_half, H]   # Top-left
# #     ])
# #     edges = [
# #         (0,1), (0,2), (0,3), (0,4),  # Apex to base
# #         (1,2), (2,3), (3,4), (4,1)   # Base rectangle
# #     ]
# #     return vertices, edges

# # # Create rotation matrix from Euler angles (yaw, pitch, roll) in degrees
# # def get_rotation_matrix(yaw, pitch, roll):
# #     return Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()

# # # Apply transformation (rotation + translation) to vertices
# # def transform_vertices(vertices, rotation_matrix, translation):
# #     transformed = (rotation_matrix @ vertices.T).T + translation
# #     return transformed

# # # Plot the 3D scene
# # def plot_scene(ax, cube_vertices, cube_edges, frustum_vertices, frustum_edges, w, h, n_subdivisions):
# #     ax.clear()  # Clear previous plot

# #     # Plot cube
# #     for edge in cube_edges:
# #         v0, v1 = cube_vertices[edge[0]], cube_vertices[edge[1]]
# #         ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 'b-', label='Cube' if edge == cube_edges[0] else "")

# #     # Plot frustum
# #     for edge in frustum_edges:
# #         v0, v1 = frustum_vertices[edge[0]], frustum_vertices[edge[1]]
# #         ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 'r-', label='Frustum' if edge == frustum_edges[0] else "")

# #     # Plot image plane as a rectangular grid with gaps
# #     w_half, h_half = w / 2, h / 2
# #     plane_vertices = frustum_vertices[1:5]  # Base vertices
# #     u = np.linspace(0, 1, n_subdivisions)
# #     v = np.linspace(0, 1, n_subdivisions)
    
# #     X = (1-u[:, None]) * (1-v[None, :]) * plane_vertices[0, 0] + u[:, None] * (1-v[None, :]) * plane_vertices[1, 0] + \
# #         (1-u[:, None]) * v[None, :] * plane_vertices[3, 0] + u[:, None] * v[None, :] * plane_vertices[2, 0]
# #     Y = (1-u[:, None]) * (1-v[None, :]) * plane_vertices[0, 1] + u[:, None] * (1-v[None, :]) * plane_vertices[1, 1] + \
# #         (1-u[:, None]) * v[None, :] * plane_vertices[3, 1] + u[:, None] * v[None, :] * plane_vertices[2, 1]
# #     Z = (1-u[:, None]) * (1-v[None, :]) * plane_vertices[0, 2] + u[:, None] * (1-v[None, :]) * plane_vertices[1, 2] + \
# #         (1-u[:, None]) * v[None, :] * plane_vertices[3, 2] + u[:, None] * v[None, :] * plane_vertices[2, 2]
    
# #     for i in range(n_subdivisions):
# #         ax.plot(X[:, i], Y[:, i], Z[:, i], 'g-', linewidth=0.5, label='Image Plane' if i == 0 else "")
# #         ax.plot(X[i, :], Y[i, :], Z[i, :], 'g-', linewidth=0.5)

# #     # Set labels and title
# #     ax.set_xlabel('X')
# #     ax.set_ylabel('Y')
# #     ax.set_zlabel('Z')
# #     ax.set_title('3D Scene: Cube and Camera Frustum')
    
# #     # Add legend
# #     ax.legend(['Cube', 'Frustum', 'Image Plane'])

# #     # Equal aspect ratio
# #     max_range = np.ptp(np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]), axis=1).max() / 2
# #     mid_x = np.mean(ax.get_xlim3d())
# #     mid_y = np.mean(ax.get_ylim3d())
# #     mid_z = np.mean(ax.get_zlim3d())
# #     ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
# #     ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
# #     ax.set_zlim3d(mid_z - max_range, mid_z + max_range)

# # # Initial parameters
# # initial_params = {
# #     'L': 1.0,
# #     'cube_tx': 2.0, 'cube_ty': 2.0, 'cube_tz': 2.0,
# #     'cube_yaw': 30.0, 'cube_pitch': 60.0, 'cube_roll': 90.0,
# #     'w': 1.0, 'h': 0.75, 'H': 1.0,
# #     'camera_tx': -2.0, 'camera_ty': 0.0, 'camera_tz': 0.5,
# #     'camera_yaw': 0.0, 'camera_pitch': 0.0, 'camera_roll': 0.0,
# #     'n_subdivisions': 10
# # }

# # # Main execution
# # if __name__ == "__main__":
# #     # Create figure and 3D axes
# #     fig = plt.figure(figsize=(12, 10))
# #     ax = fig.add_subplot(111, projection='3d')
    
# #     # Adjust layout to make space for sliders and button
# #     plt.subplots_adjust(bottom=0.55, top=0.95)
    
# #     # Define slider axes (split into two columns for better layout)
# #     slider_axes = [
# #         plt.axes([0.15, 0.02 + i*0.03, 0.35, 0.02]) for i in range(8)
# #     ] + [
# #         plt.axes([0.55, 0.02 + i*0.03, 0.35, 0.02]) for i in range(9)
# #     ]
    
# #     # Define button axis
# #     button_ax = plt.axes([0.45, 0.48, 0.1, 0.03])
    
# #     # Create sliders
# #     sliders = {
# #         'L': Slider(slider_axes[0], 'Cube L', 0.1, 5.0, valinit=initial_params['L']),
# #         'cube_tx': Slider(slider_axes[1], 'Cube Tx', -5.0, 5.0, valinit=initial_params['cube_tx']),
# #         'cube_ty': Slider(slider_axes[2], 'Cube Ty', -5.0, 5.0, valinit=initial_params['cube_ty']),
# #         'cube_tz': Slider(slider_axes[3], 'Cube Tz', -5.0, 5.0, valinit=initial_params['cube_tz']),
# #         'cube_yaw': Slider(slider_axes[4], 'Cube Yaw', -180.0, 180.0, valinit=initial_params['cube_yaw']),
# #         'cube_pitch': Slider(slider_axes[5], 'Cube Pitch', -180.0, 180.0, valinit=initial_params['cube_pitch']),
# #         'cube_roll': Slider(slider_axes[6], 'Cube Roll', -180.0, 180.0, valinit=initial_params['cube_roll']),
# #         'w': Slider(slider_axes[7], 'Frustum w', 0.1, 5.0, valinit=initial_params['w']),
# #         'h': Slider(slider_axes[8], 'Frustum h', 0.1, 5.0, valinit=initial_params['h']),
# #         'H': Slider(slider_axes[9], 'Frustum H', 0.1, 5.0, valinit=initial_params['H']),
# #         'camera_tx': Slider(slider_axes[10], 'Camera Tx', -5.0, 5.0, valinit=initial_params['camera_tx']),
# #         'camera_ty': Slider(slider_axes[11], 'Camera Ty', -5.0, 5.0, valinit=initial_params['camera_ty']),
# #         'camera_tz': Slider(slider_axes[12], 'Camera Tz', -5.0, 5.0, valinit=initial_params['camera_tz']),
# #         'camera_yaw': Slider(slider_axes[13], 'Camera Yaw', -180.0, 180.0, valinit=initial_params['camera_yaw']),
# #         'camera_pitch': Slider(slider_axes[14], 'Camera Pitch', -180.0, 180.0, valinit=initial_params['camera_pitch']),
# #         'camera_roll': Slider(slider_axes[15], 'Camera Roll', -180.0, 180.0, valinit=initial_params['camera_roll']),
# #         'n_subdivisions': Slider(slider_axes[16], 'Grid Subdiv', 5, 20, valinit=initial_params['n_subdivisions'], valstep=1)
# #     }
    
# #     # Create save button
# #     save_button = Button(button_ax, 'Save Plot')
    
# #     # Define update function
# #     def update(val):
# #         L = sliders['L'].val
# #         cube_translation = np.array([sliders['cube_tx'].val, sliders['cube_ty'].val, sliders['cube_tz'].val])
# #         cube_rotation = [sliders['cube_yaw'].val, sliders['cube_pitch'].val, sliders['cube_roll'].val]
# #         w = sliders['w'].val
# #         h = sliders['h'].val
# #         H = sliders['H'].val
# #         camera_translation = np.array([sliders['camera_tx'].val, sliders['camera_ty'].val, sliders['camera_tz'].val])
# #         camera_rotation = [sliders['camera_yaw'].val, sliders['camera_pitch'].val, sliders['camera_roll'].val]
# #         n_subdivisions = int(sliders['n_subdivisions'].val)
        
# #         cube_vertices, cube_edges = define_cube(L)
# #         frustum_vertices, frustum_edges = define_frustum(w, h, H)
        
# #         cube_rot_matrix = get_rotation_matrix(*cube_rotation)
# #         camera_rot_matrix = get_rotation_matrix(*camera_rotation)
        
# #         cube_vertices_transformed = transform_vertices(cube_vertices, cube_rot_matrix, cube_translation)
# #         frustum_vertices_transformed = transform_vertices(frustum_vertices, camera_rot_matrix, camera_translation)
        
# #         plot_scene(ax, cube_vertices_transformed, cube_edges, frustum_vertices_transformed, frustum_edges, w, h, n_subdivisions)
# #         fig.canvas.draw_idle()
    
# #     # Define save function
# #     def save_plot(event):
# #         plt.savefig('scene.png', dpi=300, bbox_inches='tight')
# #         print("Plot saved as 'scene.png'")
    
# #     # Register update function with sliders and save function with button
# #     for slider in sliders.values():
# #         slider.on_changed(update)
# #     save_button.on_clicked(save_plot)
    
# #     # Initial plot
# #     update(None)
    
# #     # Keep plot alive
# #     plt.show(block=True)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# Define cube vertices centered at (0,0,0)
def define_cube(L):
    L_half = L / 2
    vertices = np.array([
        [L_half, L_half, L_half],
        [L_half, L_half, -L_half],
        [L_half, -L_half, L_half],
        [L_half, -L_half, -L_half],
        [-L_half, L_half, L_half],
        [-L_half, L_half, -L_half],
        [-L_half, -L_half, L_half],
        [-L_half, -L_half, -L_half]
    ])
    edges = [
        (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
    ]
    return vertices, edges

# Define camera frustum vertices (vertex at origin, base at z=H)
def define_frustum(w, h, H):
    w_half, h_half = w / 2, h / 2
    vertices = np.array([
        [0, 0, 0],           # Apex (camera position)
        [w_half, h_half, H],   # Top-right
        [w_half, -h_half, H],  # Bottom-right
        [-w_half, -h_half, H], # Bottom-left
        [-w_half, h_half, H]   # Top-left
    ])
    edges = [
        (0,1), (0,2), (0,3), (0,4),  # Apex to base
        (1,2), (2,3), (3,4), (4,1)   # Base rectangle
    ]
    return vertices, edges

# Create rotation matrix from Euler angles (yaw, pitch, roll) in degrees
def get_rotation_matrix(yaw, pitch, roll):
    return Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()

# Apply transformation (rotation + translation) to vertices
def transform_vertices(vertices, rotation_matrix, translation):
    transformed = (rotation_matrix @ vertices.T).T + translation
    return transformed

# Plot the 3D scene
def plot_scene(cube_vertices, cube_edges, frustum_vertices, frustum_edges):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot cube
    for edge in cube_edges:
        v0, v1 = cube_vertices[edge[0]], cube_vertices[edge[1]]
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 'b-', label='Cube' if edge == cube_edges[0] else "")

    # Plot frustum
    for edge in frustum_edges:
        v0, v1 = frustum_vertices[edge[0]], frustum_vertices[edge[1]]
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 'r-', label='Frustum' if edge == frustum_edges[0] else "")

    # Plot XY-plane as a filled surface with grid lines
    x_range = np.linspace(-10, 10, 19)
    y_range = np.linspace(-10, 10, 19)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)  # z=0 for XY-plane
    ax.plot_surface(X, Y, Z, color='black', alpha=0.1, label='XY Plane')
    for i in range(19):
        ax.plot([x_range[0], x_range[-1]], [y_range[i], y_range[i]], [0, 0], 'black', linewidth=0.5)
        ax.plot([x_range[i], x_range[i]], [y_range[0], y_range[-1]], [0, 0], 'black', linewidth=0.5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scene: Cube, Camera Frustum, and XY Plane (Camera Aligned)')

    # Equal aspect ratio
    max_range = np.ptp(np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]), axis=1).max() / 2
    mid_x = np.mean(ax.get_xlim3d())
    mid_y = np.mean(ax.get_ylim3d())
    mid_z = np.mean(ax.get_zlim3d())
    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)

    # Commented-out annotations (as in original code)
    # ax.scatter(0, 0, 0+0.05, color='black', s=50)
    # ax.text(0, 0, 0 - 0.5, 'Origin', color='black', fontsize=10)
    # ax.scatter(camera_translation[0], camera_translation[1], camera_translation[2], color='red', s=50)
    # ax.text(camera_translation[0] - 0.5, camera_translation[1] + 0.5, camera_translation[2] - 0.5, 'Camera center', color='red', fontsize=10)
    # ax.text(cube_translation[0], cube_translation[1], cube_translation[2] - 1, 'Cube', color='blue', fontsize=10)

    ax.set_axis_off()

    plt.show()

# Parameters
L = 1  # Cube side length
cube_translation = np.array([4, 2, 1.5])  # [Tx, Ty, Tz]
cube_rotation = [30, 60, 90]  # [yaw, pitch, roll] in degrees
w, h, H = 2, 1.5, 1.5  # Width, height, distance of image plane
camera_translation = np.array([0, 0, 0])  # Camera at origin
initial_camera_rotation = [-34, 56, -51]  # Initial [yaw, pitch, roll] in degrees
target_camera_rotation = [90, 0, 0]  # Target [yaw, pitch, roll] in degrees

# Main execution
if __name__ == "__main__":
    # Define cube and frustum
    cube_vertices, cube_edges = define_cube(L)
    frustum_vertices, frustum_edges = define_frustum(w, h, H)

    # Get initial camera rotation matrix
    initial_camera_rot_matrix = get_rotation_matrix(*initial_camera_rotation)
    
    # Compute inverse of initial camera rotation to transform world to camera frame
    inverse_initial_camera_rot = np.linalg.inv(initial_camera_rot_matrix)
    
    # Compute target camera rotation matrix
    target_camera_rot_matrix = get_rotation_matrix(*target_camera_rotation)
    
    # Compute relative rotation from initial to target orientation
    relative_rotation = target_camera_rot_matrix @ inverse_initial_camera_rot

    # Transform cube: apply cube rotation, cube translation, then camera frame transform, then relative rotation
    cube_rot_matrix = get_rotation_matrix(*cube_rotation)
    cube_vertices_transformed = transform_vertices(cube_vertices, cube_rot_matrix, cube_translation)
    cube_vertices_transformed = transform_vertices(cube_vertices_transformed, inverse_initial_camera_rot, np.array([0, 0, 0]))
    cube_vertices_transformed = transform_vertices(cube_vertices_transformed, relative_rotation, np.array([0, 0, 0]))

    # Transform frustum: already in camera frame, apply relative rotation
    frustum_vertices_transformed = transform_vertices(frustum_vertices, relative_rotation, np.array([0, 0, 0]))

    # Transform XY-plane vertices for plotting
    x_range = np.linspace(-10, 10, 19)
    y_range = np.linspace(-10, 10, 19)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    xy_plane_vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    xy_plane_vertices_transformed = transform_vertices(xy_plane_vertices, relative_rotation, np.array([0, 0, 0]))
    X_transformed = xy_plane_vertices_transformed[:, 0].reshape(X.shape)
    Y_transformed = xy_plane_vertices_transformed[:, 1].reshape(Y.shape)
    Z_transformed = xy_plane_vertices_transformed[:, 2].reshape(Z.shape)

    # Update plot_scene to use transformed XY-plane
    def plot_scene(cube_vertices, cube_edges, frustum_vertices, frustum_edges):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot cube
        for edge in cube_edges:
            v0, v1 = cube_vertices[edge[0]], cube_vertices[edge[1]]
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 'b-', label='Cube' if edge == cube_edges[0] else "")

        # Plot frustum
        for edge in frustum_edges:
            v0, v1 = frustum_vertices[edge[0]], frustum_vertices[edge[1]]
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 'r-', label='Frustum' if edge == frustum_edges[0] else "")

        # Plot transformed XY-plane as a filled surface with grid lines
        ax.plot_surface(X_transformed, Y_transformed, Z_transformed, color='black', alpha=0.1, label='XY Plane')
        for i in range(19):
            ax.plot(X_transformed[:, i], Y_transformed[:, i], Z_transformed[:, i], 'black', linewidth=0.5)
            ax.plot(X_transformed[i, :], Y_transformed[i, :], Z_transformed[i, :], 'black', linewidth=0.5)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Scene: Cube, Camera Frustum, and XY Plane (Camera Aligned)')

        # Equal aspect ratio
        max_range = np.ptp(np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]), axis=1).max() / 2
        mid_x = np.mean(ax.get_xlim3d())
        mid_y = np.mean(ax.get_ylim3d())
        mid_z = np.mean(ax.get_zlim3d())
        ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
        ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
        ax.set_zlim3d(mid_z - max_range, mid_z + max_range)

        ax.set_axis_off()

        plt.show()

    # Plot the transformed scene
    plot_scene(cube_vertices_transformed, cube_edges, frustum_vertices_transformed, frustum_edges)