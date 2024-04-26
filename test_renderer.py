from scene import Scene, ManoGaussianModel
from mesh_renderer import NVDiffRenderer
from scene.cameras import MiniCam
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from PIL import Image


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
    def __init__(self, W, H, r=2, fovy=60, near=0.1, far=10):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0.1, 0.1, 1], dtype=np.float32)  # look at this point
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
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
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
        # return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]
        return np.linalg.inv(self.pose) @ self.perspective

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


def main():
    hand_model = ManoGaussianModel(sh_degree=1)
    print(hand_model.mano_model.faces.shape)
    npz_path = "output/b6e05868-e/point_cloud/iteration_1/mano_param.npz" # "output/59b3d072-1/point_cloud/iteration_2/mano_param.npz"
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

    renderer = NVDiffRenderer(use_opengl=False)
    faces = torch.from_numpy(hand_model.mano_model.faces.astype(np.int32)).to(vertices.device)

    orbit = OrbitCamera(400, 400, r=2)
    for i in range(10):
        v = vertices[0, [i]]
        print(v)
        print(renderer.world_to_camera(v.unsqueeze(0), torch.from_numpy(orbit.pose).to(vertices.device).unsqueeze(0)))
        # print(renderer.world_to_clip(v.unsqueeze(0), None, None, (400, 400), torch.from_numpy(orbit.view).to(vertices.device).unsqueeze(0)))
    # return
    # wv = torch.from_numpy(orbit.view)
    # print(wv)
    # print(orbit.pose)
    # print(orbit.perspective)
    views = []
    for ele in range(-90, 90, 30):
        for azi in range(-180, 180, 60):
            view = orbit_camera(ele, azi, 2, target=[0.1, 0.1, 1], opengl=False)
            # print(view)
            # views.append(view)
    views.append(orbit.pose)
    views.append(orbit.view)
    proj = torch.from_numpy(orbit.mvp)
    for view in views:
        print(view)
        cam = MiniCam(orbit.W, orbit.H, orbit.fovy, orbit.fovx, orbit.near, orbit.far, torch.from_numpy(view), proj, 1)

        output = renderer.render_from_camera(vertices, faces, cam, background_color=[0, 0, 0])
        rgba_mesh = output['rgba'].squeeze(0)  # (C, W, H)
        print(rgba_mesh.shape)
        image_data = rgba_mesh[:3].cpu()
        if torch.sum(image_data) < 1:
            continue
        plt.imshow(image_data)
        plt.show()


if __name__ == '__main__':
    main()
