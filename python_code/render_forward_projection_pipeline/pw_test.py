import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import tkinter as tk
from tkinter import ttk
import json

# Utility function to create rotation matrix from Euler angles (yaw, pitch, roll) in degrees
def get_rotation_matrix(yaw, pitch, roll):
    return Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()

# Utility function to transform vertices (points) with rotation and translation
def transform_vertices(vertices, rotation_matrix=None, translation=None):
    if rotation_matrix is None:
        rotation_matrix = np.eye(3)
    if translation is None:
        translation = np.zeros(3)
    try:
        return (rotation_matrix @ vertices.T).T + translation
    except ValueError as e:
        print(f"Transformation error: {e}")
        return vertices

# Cube class to manage individual cube parameters
class Cube:
    def __init__(self, size=1.0, position=[0, 0, 0], rotation=[0, 0, 0], label_offset=[0.2, 0.2, 0.2]):
        self.size = max(0.1, size)
        self.position = np.array(position)
        self.rotation = np.array(rotation)
        self.label_offset = np.array(label_offset)

    def get_data(self):
        half_size = self.size / 2
        vertices = np.array([
            [half_size, half_size, half_size], [half_size, half_size, -half_size],
            [half_size, -half_size, half_size], [half_size, -half_size, -half_size],
            [-half_size, half_size, half_size], [-half_size, half_size, -half_size],
            [-half_size, -half_size, half_size], [-half_size, -half_size, -half_size]
        ])
        edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
        rot_matrix = get_rotation_matrix(*self.rotation)
        vertices = transform_vertices(vertices, rot_matrix, self.position)
        return vertices, edges

# Scene class to manage all elements
class Scene:
    def __init__(self):
        self.cubes = [Cube(size=1.0, position=[1, 1, 1], rotation=[45, 45, 0])]
        self.frustum_width = 2.0
        self.frustum_height = 1.5
        self.frustum_depth = 1.0
        self.camera_position = np.array([-2, 0, 0])
        self.camera_rotation = np.array([0, 0, 0])
        self.points = [np.array([0, 0, 0]), np.array([1, 1, 1])]
        self.linkages = [(0, 1)]
        self.whole_scene_rotation = np.array([0, 0, 0])
        self.whole_scene_translation = np.array([0, 0, 0])
        self.show_xy_plane = True
        self.show_labels = True
        self.origin_label_offset = np.array([0.2, 0.2, 0.2])
        self.camera_label_offset = np.array([0.2, 0.2, 0.2])

    def get_frustum_data(self):
        w_half, h_half = self.frustum_width / 2, self.frustum_height / 2
        vertices = np.array([
            [0, 0, 0], [w_half, h_half, self.frustum_depth], [w_half, -h_half, self.frustum_depth],
            [-w_half, -h_half, self.frustum_depth], [-w_half, h_half, self.frustum_depth]
        ])
        edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]
        rot_matrix = get_rotation_matrix(*self.camera_rotation)
        vertices = transform_vertices(vertices, rot_matrix, self.camera_position)
        return vertices, edges

    def get_xy_plane_data(self):
        x_range = np.linspace(-10, 10, 19)
        y_range = np.linspace(-10, 10, 19)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        xy_plane_vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        return X, Y, Z, xy_plane_vertices

# Function to project points onto the camera's image plane
def project_to_image_plane(vertices, camera_position, camera_rotation, frustum_depth):
    rot_matrix = get_rotation_matrix(*camera_rotation)
    inv_rot = np.linalg.inv(rot_matrix)
    vertices_camera = transform_vertices(vertices - camera_position, inv_rot, np.zeros(3))
    projected = []
    for v in vertices_camera:
        if v[2] > 0:  # Points in front of camera
            scale = frustum_depth / v[2]
            projected.append([v[0] * scale, v[1] * scale])
        else:
            projected.append([np.nan, np.nan])  # Behind camera
    return np.array(projected)

# Function to plot the 3D scene
def plot_3d_scene(ax, scene):
    ax.clear()
    whole_rot_matrix = get_rotation_matrix(*scene.whole_scene_rotation)
    whole_trans = scene.whole_scene_translation

    # Cubes
    for i, cube in enumerate(scene.cubes):
        vertices, edges = cube.get_data()
        vertices = transform_vertices(vertices, whole_rot_matrix, whole_trans)
        for edge in edges:
            v0, v1 = vertices[edge[0]], vertices[edge[1]]
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 'b-', linewidth=2)
        if scene.show_labels:
            center = np.mean(vertices, axis=0)
            ax.text(*(center + cube.label_offset), f'Cube {i+1}', fontsize=8)

    # Frustum
    frustum_vertices, frustum_edges = scene.get_frustum_data()
    frustum_vertices = transform_vertices(frustum_vertices, whole_rot_matrix, whole_trans)
    for edge in frustum_edges:
        v0, v1 = frustum_vertices[edge[0]], frustum_vertices[edge[1]]
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 'r-', linewidth=2)

    # Points and linkages
    if scene.points:
        points = transform_vertices(np.array(scene.points), whole_rot_matrix, whole_trans)
        for point in points:
            ax.scatter(point[0], point[1], point[2], color='green', s=50)
        for link in scene.linkages:
            p0, p1 = points[link[0]], points[link[1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k--', linewidth=1)
        if scene.show_labels:
            ax.text(*(points[0] + scene.origin_label_offset), 'Origin', fontsize=8)
            ax.text(*(frustum_vertices[0] + scene.camera_label_offset), 'Camera', fontsize=8)

    # XY-plane
    if scene.show_xy_plane:
        X, Y, Z, xy_plane_vertices = scene.get_xy_plane_data()
        xy_plane_vertices = transform_vertices(xy_plane_vertices, whole_rot_matrix, whole_trans)
        X_transformed = xy_plane_vertices[:, 0].reshape(X.shape)
        Y_transformed = xy_plane_vertices[:, 1].reshape(Y.shape)
        Z_transformed = xy_plane_vertices[:, 2].reshape(Z.shape)
        ax.plot_surface(X_transformed, Y_transformed, Z_transformed, color='black', alpha=0.1)
        for i in range(19):
            ax.plot(X_transformed[:, i], Y_transformed[:, i], Z_transformed[:, i], 'black', linewidth=0.5)
            ax.plot(X_transformed[i, :], Y_transformed[i, :], Z_transformed[i, :], 'black', linewidth=0.5)

    ax.set_axis_off()

# Function to plot the camera view (2D projection)
def plot_camera_view(ax, scene):
    ax.clear()
    whole_rot_matrix = get_rotation_matrix(*scene.whole_scene_rotation)
    whole_trans = scene.whole_scene_translation

    # Project cubes
    for cube in scene.cubes:
        vertices, edges = cube.get_data()
        vertices = transform_vertices(vertices, whole_rot_matrix, whole_trans)
        projected = project_to_image_plane(vertices, scene.camera_position, scene.camera_rotation, scene.frustum_depth)
        for edge in edges:
            p0, p1 = projected[edge[0]], projected[edge[1]]
            if not (np.isnan(p0).any() or np.isnan(p1).any()):
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'b-', linewidth=2)

    # Project points and linkages
    if scene.points:
        points = transform_vertices(np.array(scene.points), whole_rot_matrix, whole_trans)
        projected_points = project_to_image_plane(points, scene.camera_position, scene.camera_rotation, scene.frustum_depth)
        for p in projected_points:
            if not np.isnan(p).any():
                ax.scatter(p[0], p[1], color='green', s=50)
        for link in scene.linkages:
            p0, p1 = projected_points[link[0]], projected_points[link[1]]
            if not (np.isnan(p0).any() or np.isnan(p1).any()):
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k--', linewidth=1)

    ax.set_xlim(-scene.frustum_width/2, scene.frustum_width/2)
    ax.set_ylim(-scene.frustum_height/2, scene.frustum_height/2)
    ax.set_aspect('equal')
    ax.set_axis_off()

# Function to print scene configuration
def print_scene_config(scene):
    config = {
        'cubes': [{
            'size': float(cube.size),
            'position': cube.position.tolist(),
            'rotation': cube.rotation.tolist(),
            'label_offset': cube.label_offset.tolist()
        } for cube in scene.cubes],
        'frustum': {
            'width': float(scene.frustum_width),
            'height': float(scene.frustum_height),
            'depth': float(scene.frustum_depth)
        },
        'camera': {
            'position': scene.camera_position.tolist(),
            'rotation': scene.camera_rotation.tolist()
        },
        'points': [p.tolist() for p in scene.points],
        'linkages': scene.linkages,
        'whole_scene': {
            'rotation': scene.whole_scene_rotation.tolist(),
            'translation': scene.whole_scene_translation.tolist()
        },
        'show_xy_plane': scene.show_xy_plane,
        'show_labels': scene.show_labels,
        'origin_label_offset': scene.origin_label_offset.tolist(),
        'camera_label_offset': scene.camera_label_offset.tolist()
    }
    print("\nScene Configuration:")
    print(json.dumps(config, indent=2))

# GUI for parameter tuning
class ParameterGUI:
    def __init__(self, scene, fig_3d, ax_3d, fig_camera, ax_camera):
        self.scene = scene
        self.fig_3d = fig_3d
        self.ax_3d = ax_3d
        self.fig_camera = fig_camera
        self.ax_camera = ax_camera
        self.root = tk.Tk()
        self.root.title("Parameter Tuning")
        self.root.geometry("400x600")

        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.current_cube_idx = tk.StringVar(value="0")
        self.sliders = {}
        self.checkboxes = {}
        self.create_gui()

    def create_gui(self):
        # Cube selection
        ttk.Label(self.scroll_frame, text="Select Cube:").pack()
        cube_options = [str(i) for i in range(len(self.scene.cubes))]
        cube_dropdown = ttk.OptionMenu(self.scroll_frame, self.current_cube_idx, cube_options[0], *cube_options, command=self.update_sliders)
        cube_dropdown.pack()
        ttk.Button(self.scroll_frame, text="Add Cube", command=self.add_cube).pack()

        # Parameters
        params = [
            ('Cube Size', 'cube_size', 0.1, 5.0, lambda c: c.size),
            ('Cube X', 'cube_position', -5, 5, lambda c: c.position[0], 0),
            ('Cube Y', 'cube_position', -5, 5, lambda c: c.position[1], 1),
            ('Cube Z', 'cube_position', -5, 5, lambda c: c.position[2], 2),
            ('Cube Yaw', 'cube_rotation', -180, 180, lambda c: c.rotation[0], 0),
            ('Cube Pitch', 'cube_rotation', -180, 180, lambda c: c.rotation[1], 1),
            ('Cube Roll', 'cube_rotation', -180, 180, lambda c: c.rotation[2], 2),
            ('Cube Label X', 'cube_label_offset', -1, 1, lambda c: c.label_offset[0], 0),
            ('Cube Label Y', 'cube_label_offset', -1, 1, lambda c: c.label_offset[1], 1),
            ('Cube Label Z', 'cube_label_offset', -1, 1, lambda c: c.label_offset[2], 2),
            ('Frustum Width', 'frustum_width', 0.1, 5.0, lambda s: s.frustum_width),
            ('Frustum Height', 'frustum_height', 0.1, 5.0, lambda s: s.frustum_height),
            ('Frustum Depth', 'frustum_depth', 0.1, 5.0, lambda s: s.frustum_depth),
            ('Camera X', 'camera_position', -5, 5, lambda s: s.camera_position[0], 0),
            ('Camera Y', 'camera_position', -5, 5, lambda s: s.camera_position[1], 1),
            ('Camera Z', 'camera_position', -5, 5, lambda s: s.camera_position[2], 2),
            ('Camera Yaw', 'camera_rotation', -180, 180, lambda s: s.camera_rotation[0], 0),
            ('Camera Pitch', 'camera_rotation', -180, 180, lambda s: s.camera_rotation[1], 1),
            ('Camera Roll', 'camera_rotation', -180, 180, lambda s: s.camera_rotation[2], 2),
            ('Whole X', 'whole_scene_translation', -5, 5, lambda s: s.whole_scene_translation[0], 0),
            ('Whole Y', 'whole_scene_translation', -5, 5, lambda s: s.whole_scene_translation[1], 1),
            ('Whole Z', 'whole_scene_translation', -5, 5, lambda s: s.whole_scene_translation[2], 2),
            ('Whole Yaw', 'whole_scene_rotation', -180, 180, lambda s: s.whole_scene_rotation[0], 0),
            ('Whole Pitch', 'whole_scene_rotation', -180, 180, lambda s: s.whole_scene_rotation[1], 1),
            ('Whole Roll', 'whole_scene_rotation', -180, 180, lambda s: s.whole_scene_rotation[2], 2),
            ('Origin Label X', 'origin_label_offset', -1, 1, lambda s: s.origin_label_offset[0], 0),
            ('Origin Label Y', 'origin_label_offset', -1, 1, lambda s: s.origin_label_offset[1], 1),
            ('Origin Label Z', 'origin_label_offset', -1, 1, lambda s: s.origin_label_offset[2], 2),
            ('Camera Label X', 'camera_label_offset', -1, 1, lambda s: s.camera_label_offset[0], 0),
            ('Camera Label Y', 'camera_label_offset', -1, 1, lambda s: s.camera_label_offset[1], 1),
            ('Camera Label Z', 'camera_label_offset', -1, 1, lambda s: s.camera_label_offset[2], 2),
        ]
        for i, point in enumerate(self.scene.points):
            params.extend([
                (f'Point {i+1} X', 'points', -5, 5, lambda s: s.points[i][0], i, 0),
                (f'Point {i+1} Y', 'points', -5, 5, lambda s: s.points[i][1], i, 1),
                (f'Point {i+1} Z', 'points', -5, 5, lambda s: s.points[i][2], i, 2),
            ])

        for label, attr, min_val, max_val, get_val, *idx in params:
            frame = ttk.Frame(self.scroll_frame)
            frame.pack(fill='x', pady=2)
            ttk.Label(frame, text=label).pack(side='left')
            var = tk.DoubleVar(value=float(get_val(self.scene)))
            scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, command=lambda v, l=label, a=attr, i=idx: self.update_param(l, a, i, float(v)))
            scale.pack(side='right', fill='x', expand=True)
            self.sliders[label] = (var, scale)

        # Checkboxes
        ttk.Label(self.scroll_frame, text="Toggles:").pack()
        self.checkboxes['Show XY Plane'] = tk.BooleanVar(value=self.scene.show_xy_plane)
        ttk.Checkbutton(self.scroll_frame, text="Show XY Plane", variable=self.checkboxes['Show XY Plane'], command=self.update_toggles).pack()
        self.checkboxes['Show Labels'] = tk.BooleanVar(value=self.scene.show_labels)
        ttk.Checkbutton(self.scroll_frame, text="Show Labels", variable=self.checkboxes['Show Labels'], command=self.update_toggles).pack()

    def update_param(self, label, attr, idx, value):
        cube_idx = int(self.current_cube_idx.get())
        if 'cube_' in attr:
            cube = self.scene.cubes[cube_idx]
            if len(idx) == 1:
                getattr(cube, attr)[idx[0]] = value
            else:
                setattr(cube, attr, value)
        elif len(idx) == 2:
            self.scene.points[idx[0]][idx[1]] = value
        elif len(idx) == 1:
            getattr(self.scene, attr)[idx[0]] = value
        else:
            setattr(self.scene, attr, value)
        self.update_plots()
        print_scene_config(self.scene)

    def update_toggles(self):
        self.scene.show_xy_plane = self.checkboxes['Show XY Plane'].get()
        self.scene.show_labels = self.checkboxes['Show Labels'].get()
        self.update_plots()
        print_scene_config(self.scene)

    def add_cube(self):
        self.scene.cubes.append(Cube())
        cube_options = [str(i) for i in range(len(self.scene.cubes))]
        self.current_cube_idx.set(str(len(self.scene.cubes)-1))
        self.root.destroy()
        self.__init__(self.scene, self.fig_3d, self.ax_3d, self.fig_camera, self.ax_camera)
        self.update_plots()
        print_scene_config(self.scene)

    def update_sliders(self, *args):
        cube_idx = int(self.current_cube_idx.get())
        for label, (var, scale) in self.sliders.items():
            if 'Cube ' in label:
                parts = label.split()
                attr = 'cube_' + parts[1].lower() if parts[1] in ['Size', 'X', 'Y', 'Z'] else 'cube_' + parts[1].lower() + '_offset'
                idx = {'X': 0, 'Y': 1, 'Z': 2, 'Yaw': 0, 'Pitch': 1, 'Roll': 2}.get(parts[1], None)
                if idx is not None:
                    value = getattr(self.scene.cubes[cube_idx], attr)[idx]
                else:
                    value = getattr(self.scene.cubes[cube_idx], attr)
                var.set(float(value))
        self.update_plots()
        print_scene_config(self.scene)

    def update_plots(self):
        plot_3d_scene(self.ax_3d, self.scene)
        plot_camera_view(self.ax_camera, self.scene)
        self.fig_3d.canvas.draw_idle()
        self.fig_camera.canvas.draw_idle()

# Main execution
if __name__ == "__main__":
    scene = Scene()
    fig_3d = plt.figure(figsize=(8, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    fig_camera = plt.figure(figsize=(4, 4))
    ax_camera = fig_camera.add_subplot(111)
    gui = ParameterGUI(scene, fig_3d, ax_3d, fig_camera, ax_camera)
    plot_3d_scene(ax_3d, scene)
    plot_camera_view(ax_camera, scene)
    print_scene_config(scene)
    plt.show()
    gui.root.mainloop()