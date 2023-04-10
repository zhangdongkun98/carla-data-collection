
import numpy as np
import open3d

from . import params


def save_pcd(file_path, pcd):
    np.save(file_path, pcd)

def load_pcd(file_path):
    return np.load(file_path)



def save_bbx(file_path, bbx):
    np.save(file_path, bbx)

def load_bbx(file_path):
    return np.load(file_path)





class Vis(object):
    def __init__(self):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name='title')

        render_option = self.vis.get_render_option()
        background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
        render_option.background_color = background_color
        render_option.point_color_option = open3d.visualization.PointColorOption.ZCoordinate

        self.vis_bounding_boxs = [open3d.geometry.LineSet() for _ in range(params.num_vehicles)]
        [self.vis.add_geometry(bounding_box) for bounding_box in self.vis_bounding_boxs]


    def run_step(self, obstacle_bbx_vecs, points):
        if not hasattr(self, 'pcd'):
            ### point cloud
            self.pcd = open3d.geometry.PointCloud()
            self.pcd.points = open3d.utility.Vector3dVector(points)
            self.vis.add_geometry(self.pcd)   

        number_min = min(len(obstacle_bbx_vecs), params.num_vehicles)
        number_max = max(len(obstacle_bbx_vecs), params.num_vehicles)
        for i in range(number_min):
            obstacle_bbx_vec, bounding_box = obstacle_bbx_vecs[i], self.vis_bounding_boxs[i]
            new_bounding_box = calculate_vis_bbx3d(obstacle_bbx_vec)
            bounding_box.points = new_bounding_box.points
            bounding_box.lines = new_bounding_box.lines
            bounding_box.colors = new_bounding_box.colors

        for i in range(number_min, number_max):
            self.vis_bounding_boxs[i].clear()

        self.pcd.points = open3d.utility.Vector3dVector(points)

        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()


    def destroy(self):
        self.vis.destroy_window()



def get_vehicle_bbx_vec(vehicle):
    t = vehicle.get_transform()
    bbx = vehicle.bounding_box.extent *2
    if vehicle.type_id == 'vehicle.kawasaki.ninja' or vehicle.type_id == 'vehicle.yamaha.yzf' or vehicle.type_id == 'vehicle.harley-davidson.low_rider':  ### motor
        car_type = 2
    elif vehicle.type_id == 'vehicle.gazelle.omafiets' or vehicle.type_id == 'vehicle.bh.crossbike' or vehicle.type_id == 'vehicle.diamondback.century':  ### bike
        car_type = 3
    else:  ### car
        car_type = 1
    return np.array([
        t.location.x, t.location.y, t.location.z - params.lidar_z + bbx.z/2,
        np.deg2rad(t.rotation.yaw),
        bbx.x, bbx.y, bbx.z,
        car_type,
    ], dtype=np.float32)




def calculate_vis_bbx3d(bbx_vec):
    translation = bbx_vec[:3]
    yaw = bbx_vec[3]
    l, w, h = bbx_vec[4:7]

    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = (np.dot(rotation_matrix, bounding_box) + eight_points.transpose()).T

    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
            [4, 5], [5, 6], [6, 7], [4, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    # colors = [[1, 0, 0] for _ in range(len(lines))]
    color = np.array([190,190,190]).astype(np.float64) / 255
    colors = np.expand_dims(color, axis=0).repeat(len(lines), axis=0)

    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(corner_box)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set

