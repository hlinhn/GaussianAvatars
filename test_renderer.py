from scene import Scene, ManoGaussianModel
from mesh_renderer import NVDiffRenderer
from scene.cameras import MiniCam
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from PIL import Image
import nvdiffrast.torch as dr
import os
import json
from utils.graphics_utils import focal2fov


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=False):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100.):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        x = np.tan(self.fovx / 2)
        # aspect = self.W / self.H
        return np.array(
            [
                [1 / y, 0, 0, 0],
                [0, 1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]
        # return np.linalg.inv(self.pose) @ self.perspective

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])


def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0],
                     [0,  c, -s, 0],
                     [0,  s, c, 0],
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)

def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0,  n/x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int):
    # print(pos)
    # print(mtx)
    pos_clip    = transform_pos(mtx, pos)
    # print(pos_clip)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    print(torch.sum(rast_out[..., -1:]))
    # print(rast_out.shape)
    # print(torch.sum(rast_out))
    color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color

def cube_visualization():
    datadir = '/home/halinh/projects/nvdiffrast/samples/data'
    fn = 'cube_%s.npz' % ('c')
    with np.load(f'{datadir}/{fn}') as f:
        pos_idx, vtxp, col_idx, vtxc = f.values()
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], vtxp.shape[0]))

    pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda()
    col_idx = torch.from_numpy(col_idx.astype(np.int32)).cuda()
    vtx_pos = torch.from_numpy(vtxp.astype(np.float32)).cuda()
    vtx_col = torch.from_numpy(vtxc.astype(np.float32)).cuda()

    a_rot = np.matmul(rotate_x(-0.4), rotate_y(0.0))
    proj  = projection(x=0.4)
    a_mv  = np.matmul(translate(0, 0, -3.5), a_rot)
    a_mvp = np.matmul(proj, a_mv).astype(np.float32)
    renderer = NVDiffRenderer(use_opengl=False)
    # glctx = dr.RasterizeCudaContext()
    img_d = render(renderer.glctx, a_mvp, vtx_pos, pos_idx, vtx_col, col_idx, 512)[0]
    print(img_d.shape)
    print(torch.sum(img_d))
    # plt.imshow(img_d.cpu())
    # plt.show()

    #a_mv = a_mv.T
    #a_mv[:, 1] = -a_mv[:, 1]
    #a_mv[:, 2] = -a_mv[:, 2]
    #a_mvp = a_mvp.T
    #a_mvp[:, 1] = -a_mvp[:, 1]
    orbit = OrbitCamera(512, 512, r=3.5, fovy=23)
    cam = MiniCam(orbit.W, orbit.H, orbit.fovy, orbit.fovx, orbit.near, orbit.far, 
                  torch.from_numpy(a_mv), torch.from_numpy(a_mvp), 1)
    print(pos_idx.shape)
    #output = renderer.render_from_camera(vtx_pos.unsqueeze(0), 
    #                                     pos_idx, cam, background_color=[1, 1, 1])
    output = renderer.render_mesh(vtx_pos.unsqueeze(0), 
                                  pos_idx, torch.from_numpy(a_mv).to(vtx_pos.device).unsqueeze(0), 
                                  torch.from_numpy(a_mvp).to(vtx_pos.device).unsqueeze(0),
                                  (512, 512), background_color=[1, 1, 1])
    rgba_mesh = output['rgba'].squeeze(0)  # (C, W, H)
    print(rgba_mesh.shape)
    image_data = rgba_mesh[:, :, :3].cpu()
    print(torch.sum(image_data))
    plt.imshow(image_data)
    plt.show()


def read_example_camera(camera_file):
    path = "."
    camera_infos = {}
    with open(os.path.join(path, camera_file)) as camera_json:
        contents = json.load(camera_json)
        cameras = contents['cameras']
        for c in cameras:
            transform_mat = np.eye(4)
            transform_mat[:3, :3] = np.array(c['camrot'])
            transform_mat[:3, 3] = np.array(c['campos'])
            # transform_mat[:3, 1:3] *= -1
            # w2c = np.linalg.inv(transform_mat)
            # R = np.transpose(w2c[:3, :3])
            # T = w2c[:3, 3] # np.matmul(-R.transpose(), np.array(c['campos']))
            R = transform_mat[:3, :3]
            T = transform_mat[:3, 3]
            focal = np.array(c['focal'])
            cam_id = c['id']
            camera_infos[cam_id] = {'R': R, 'T': T, 'focal': focal}
    return camera_infos


def visualize_cam_and_obj(cam_pos, cam_rot, obj):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    ax.scatter()


def main():
    #cube_visualization()
    #return
    infos = read_example_camera("test_cam_try.json")
    for k, v in infos.items():
        print(f"{k}")
    test_cam = infos["cam400479"]
    w = 334
    h = 512
    test_fovx = focal2fov(test_cam['focal'][0], w)
    test_fovy = focal2fov(test_cam['focal'][1], h)

    hand_model = ManoGaussianModel(sh_degree=1)
    print(hand_model.mano_model.faces.shape)
    npz_path = "output/59b3d072-1/point_cloud/iteration_2/mano_param.npz"
    mano_param = np.load(str(npz_path))
    mano_param = {k: torch.from_numpy(v).cuda() for k, v in mano_param.items()}
    for k, v in mano_param.items():
        print(f"{k}: {v.shape}")
    hand_output = hand_model.mano_model(
        mano_param['betas'][None],
        mano_param['global_orient'][[0]],
        mano_param['hand_pose'][[0]],
        mano_param['transl'][[0]]
    )
    vertices = hand_output.vertices
    print(vertices.shape)

    if False:
        vertices_cpu = vertices.cpu()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(vertices_cpu[0, :, 0], vertices_cpu[0, :, 1], vertices_cpu[0, :, 2])
        plt.show()
        return

    orbit = OrbitCamera(w, h, r=1.5, fovy=np.rad2deg(test_fovy)) # OrbitCamera(512, 512, r=1.5, fovy=23)
    print(orbit.fovy)
    y = np.tan(orbit.fovy / 2)
    print(y)
    print(1 / y)
    renderer = NVDiffRenderer(use_opengl=False)
    faces = torch.from_numpy(hand_model.mano_model.faces.astype(np.int32)).to(vertices.device)
    # -0.82042429 -2.63014416
    #a_rot = np.matmul(rotate_x(-2.63014416), rotate_y(-0.82042429))
    #print(a_rot)
    combi = [[1, 1, 1],
             [-1, 1, 1],
             [-1, -1, 1],
             [-1, 1, -1],
             [-1, -1, -1],
             [1, -1, 1],
             [1, 1, -1],
             [1, -1, -1]]
    proj = orbit.perspective
    print(proj)
    """
    aim_rot = R.from_matrix(test_cam['R']).inv()
    rpy = aim_rot.as_euler('zyx', False)
    print(rpy)
    a_mv = np.eye(4)
    # print(test_cam['R'])
    # print(test_cam['T'])
    a_mv[:3, :3] = test_cam['R']
    a_mv[:3, 3] = test_cam['T'] * 0.001
    a_mv = np.linalg.inv(a_mv)
    print(a_mv)
    return    
    """
    for c in combi:
        a_rot = np.eye(4)
        orig_rot = [-3.0428223, -0.82042429, -2.63014416]
        # orig_rot = [2.7625822, 0.75185267, -2.52200328]
        new_rot = orig_rot
        for i in range(3):
            new_rot[i] = new_rot[i] * c[i]
        a_rot[:3, :3] = R.from_euler('zyx', new_rot).as_matrix()
        # print(a_rot)
        # proj  = projection(x=0.001, n=0.01, f=100.0)
        # a_mv  = np.matmul(translate(-0.5, 0.0, -2.5), a_rot)
        for c2 in combi:
            orig_trans = [-0.793, -0.489, 1.737]
            # orig_trans = [-1.85720151, -0.2417274, 0.61618364]
            new_trans = orig_trans
            for i in range(3):
                new_trans[i] = new_trans[i] * c2[i]
            a_mv = np.matmul(translate(new_trans[0], new_trans[1], new_trans[2]), a_rot)
            #a_mv = np.matmul(translate(test_cam['T'][0] * -0.001, 
            #                           test_cam['T'][1] * 0.001,
            #                           test_cam['T'][2] * -0.001), a_rot)
            print('-------------')
            print(c)
            print(c2)
            print(a_mv)
            # a_mv = np.eye(4)
            # print(test_cam['R'])
            # print(test_cam['T'])
            #a_mv[:3, :3] = test_cam['R']
            #a_mv[:3, 3] = test_cam['T'] * 0.001
            # a_mv = np.linalg.inv(a_mv)
            # print(a_mv)
            a_mvp = np.matmul(proj, a_mv).astype(np.float32)
            # aim_rot = R.from_matrix(test_cam['R'])
            # rpy = aim_rot.as_euler('zyx', False)
            # print(rpy)
            # 2.52200328 -0.75185267 -2.7625822
            a_mv = a_mv.T
            a_mv[:, 1] = -a_mv[:, 1]
            a_mv[:, 2] = -a_mv[:, 2]
            a_mvp = a_mvp.T
            a_mvp[:, 1] = -a_mvp[:, 1]
            print(a_mv)
            print(a_mvp)
            cam = MiniCam(orbit.W, orbit.H, orbit.fovy, orbit.fovx, orbit.near, orbit.far, 
                        torch.from_numpy(a_mv), torch.from_numpy(a_mvp), 1)

            # orbit = OrbitCamera(400, 400, r=4, fovy=23)
            # proj  = util.projection(x=0.4)
            # a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))
            # a_mv  = np.matmul(util.translate(0, 0, -3.5), a_rot)
            # a_mvp = np.matmul(proj, a_mv).astype(np.float32)

            # for i in range(10):
            #     v = vertices[0, [-i]]
            #     print(v)
                # print(renderer.world_to_camera(v.unsqueeze(0), torch.from_numpy(orbit.pose).to(vertices.device).unsqueeze(0)))
            #     print(renderer.world_to_clip(v.unsqueeze(0), None, None, (400, 400), torch.from_numpy(orbit.mvp).to(vertices.device).unsqueeze(0)))
            # return
            # wv = torch.from_numpy(orbit.view)
            # print(wv)
            # print(orbit.pose)
            # print(orbit.perspective)
            # views = []
            # for ele in range(-90, 90, 30):
            #    for azi in range(-180, 180, 60):
            #         view = orbit_camera(ele, azi, 2, target=[0.1, 0.1, 1], opengl=False)
            #         # print(view)
            #         # views.append(view)
            # views.append(orbit.pose)
            # views.append(orbit.view)
            # proj = torch.from_numpy(orbit.mvp)
            # for view in views:
            #     print(view)
            #     cam = MiniCam(orbit.W, orbit.H, orbit.fovy, orbit.fovx, orbit.near, orbit.far, 
            #                   torch.from_numpy(view), proj, 1)

            output = renderer.render_from_camera(vertices, faces, cam, background_color=[1, 1, 1])
            rgba_mesh = output['rgba'].squeeze(0)  # (C, W, H)
            print(rgba_mesh.shape)
            image_data = rgba_mesh[:, :, :3].cpu()
            if torch.sum(image_data) >= 3 * 512 * 334:
                continue
            plt.imshow(image_data)
            plt.show()


if __name__ == '__main__':
    main()
