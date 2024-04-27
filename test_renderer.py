from scene import Scene, ManoGaussianModel
from mesh_renderer import NVDiffRenderer
from scene.cameras import MiniCam, Camera
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from PIL import Image
import nvdiffrast.torch as dr
import os
import json
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2


def make_perspective(fovy, fovx, near, far):
    y = np.tan(fovy / 2)
    x = np.tan(fovx / 2)

    return np.array(
        [
            [1 / x, 0, 0, 0],
            [0, 1 / y, 0, 0],
            [
                0,
                0,
                -(far + near) / (far - near),
                -(2 * far * near) / (far - near),
            ],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )


def read_example_camera(camera_file):
    path = "."
    camera_infos = {}
    w = 334
    h = 512
    near = 0.01
    far = 100
    with open(os.path.join(path, camera_file)) as camera_json:
        contents = json.load(camera_json)
        cameras = contents['cameras']
        for c in cameras:
            focal = np.array(c['focal'])
            fovx = focal2fov(focal[0], w)
            fovy = focal2fov(focal[1], h)
            proj = make_perspective(fovy, fovx, near, far)
            transform_mat = np.eye(4)
            transform_mat[:3, :3] = np.array(c['camrot'])
            R = transform_mat[:3, :3]
            transform_mat[:3, 3] = np.matmul(-R, np.array(c['campos'])) * 0.001
            transform_mat[1, :] = -transform_mat[1, :]
            transform_mat[2, :] = -transform_mat[2, :]
            R = transform_mat[:3, :3].transpose()
            T = transform_mat[:3, 3]
            a_mvp = np.matmul(proj, transform_mat).astype(np.float32)
            supplied_mv = transform_mat
            #supplied_mv = supplied_mv.T
            #supplied_mv[:, 1] = -supplied_mv[:, 1]
            #supplied_mv[:, 2] = -supplied_mv[:, 2]
            supplied_mvp = a_mvp
            #supplied_mvp = supplied_mvp.T
            #supplied_mvp[:, 1] = -supplied_mvp[:, 1]

            cam_id = c['id']
            camera_infos[cam_id] = {'R': R, 'T': T, 'focal': focal,
                                    'w': w, 'h': h, 'fovx': fovx, 'fovy': fovy,
                                    'mv': supplied_mv, 'mvp': supplied_mvp,
                                    'near': near, 'far': far}
    return camera_infos


def cam_info_to_minicam(expected_info):
    # in remote_viewer
    # view_matrix: world_view_transform.T
    # view_projection_matrix: full_proj_transform.T
    # in network_gui
    # world_view_transform flip 1, 2
    # full_proj_transform flip 1
    # world_view_transform = getWorld2view2(R, T).transpose(0, 1) <- transform_mat.T
    # projection_matrix = getProjectionMatrix(...).transpose(0, 1)
    # full_proj_transform = world * matrix <- a_mvp.T
    # their projection matrix also has a flip sign
    # unless reshape transpose it again - test?

    # cam_info = Camera(colmap_id=1, R=expected_info['R'], T=expected_info['T'],
    #                   FoVx=expected_info['fovx'], FoVy=expected_info['fovy'],
    #                   image_width=expected_info['w'], image_height=expected_info['h'],
    #                   bg=[1, 1, 1], image=np.ones(334, 512, 3), image_path="",
    #                   image_name="", uid=1, timestep=1, data_device='cuda')

    expected_mvp = expected_info['mvp']
    expected_transform = torch.from_numpy(np.array(expected_info['mv']))
    calculated_transform = getWorld2View2(np.array(expected_info['R']), np.array(expected_info['T']))
    print(expected_transform)
    print(calculated_transform)
    center = expected_transform.transpose(0, 1).inverse()[3, :3]
    print(center)
    other_proj = getProjectionMatrix(expected_info['near'], expected_info['far'],
                                     expected_info['fovx'], expected_info['fovy']).double()
    print(other_proj.shape)
    calculated_mvp = (expected_transform.transpose(0, 1).unsqueeze(0).bmm(other_proj.transpose(0, 1).unsqueeze(0))).squeeze(0)
    print(calculated_mvp)
    print(expected_mvp)
    calculated_flatten = calculated_mvp.T.flatten().tolist()
    received = torch.reshape(torch.tensor(calculated_flatten), (4, 4))
    print(received)


# we want to retrieve the camera in COLMAP convention (y down, z in, x right?)
# projection from intrinsics seems to think our camera space is right, up, out
# in NVDiffRenderer (render_from_camera), we receive the world to camera transform and then flip the y and z. We also flip y of mvp matrix
# after rendering we flip back the y axis of rendered image
# assuming that the camera convention we are rendering from is OpenGL (right, up, out), the received camera convention should be (right, down, in)
# for interhand, world to camera is R * mesh - Rt
# Interhand mesh renderer uses Pytorch convention (left, up, in), where they flip x and y axis of the mesh (in cam coord I assume)
# therefore the mesh is already in (right, down, in) after transformation
# therefore the world to camera matrix is already correct?
# a variable is to send the rotation transposed or not
# since all the transform in mesh renderer use points * mat.T, we can assume that the mat.T here is row major (translation in the last row). Meaning mat is column major (trans is the last col)
# therefore we only need to supply [R | -Rt], the other function will transpose it for us
# the mvp also has to be calculated with this matrix


def main():
    # check 422
    hand_model = ManoGaussianModel(sh_degree=1)
    print(hand_model.mano_model.faces.shape)
    # npz_path = "output/59b3d072-1/point_cloud/iteration_2/mano_param.npz"
    npz_path = "output/b6e05868-e/point_cloud/iteration_1/mano_param.npz"
    mano_param = np.load(str(npz_path))
    mano_param = {k: torch.from_numpy(v).cuda() for k, v in mano_param.items()}
    hand_output = hand_model.mano_model(
        mano_param['betas'][None],
        mano_param['global_orient'][[0]],
        mano_param['hand_pose'][[0]],
        mano_param['transl'][[0]]
    )
    vertices = hand_output.vertices
    # print(vertices.shape)

    renderer = NVDiffRenderer(use_opengl=False)
    faces = torch.from_numpy(hand_model.mano_model.faces.astype(np.int32)).to(vertices.device)
    infos = read_example_camera("test_cam_try.json")
    for k, v in infos.items():
        cam_info_to_minicam(v)
        exit(0)
        print(f"{k}")
        test_cam = infos[k]
        cam = MiniCam(test_cam['w'], test_cam['h'], test_cam['fovy'], test_cam['fovx'],
                      test_cam['near'], test_cam['far'],
                      torch.from_numpy(np.array(test_cam['mv'])),
                      torch.from_numpy(np.array(test_cam['mvp'])), 1)
        # a_rot = np.eye(4)
        # a_rot[:3, :3] = test_cam['R']
        # for c2 in combi:
            # trans = np.matmul(-a_rot[:3, :3], test_cam['T']) * 0.001
            # print(trans)
            # a_mv = np.matmul(translate(trans[0], trans[1], trans[2]), a_rot)
            # mesh_points = vertices[[0], :10, :]
            # output as seen from interhand renderer (world to camera)
            # mesh_interhand = np.dot(test_cam['R'], mesh_points[0].transpose(1,0).cpu()).transpose(1,0) + trans.reshape(1,3)
            # print(mesh_interhand)
            # output as seen from our renderer (world to camera)
            # a_mv[1, :] = -a_mv[1, :]
            # a_mv[2, :] = -a_mv[2, :]
            # mesh_ga = renderer.world_to_camera(mesh_points.double(), torch.from_numpy(a_mv).to(mesh_points.device).unsqueeze(0))
            # print(mesh_ga)
            # print('-------------')
            # print(a_mv)
            # a_mvp = np.matmul(proj, a_mv).astype(np.float32)
            # supplied_mv = a_mv
            # supplied_mv = supplied_mv.T
            # supplied_mv[:, 1] = -supplied_mv[:, 1]
            # supplied_mv[:, 2] = -supplied_mv[:, 2]
            # supplied_mvp = a_mvp
            # supplied_mvp = supplied_mvp.T
            # supplied_mvp[:, 1] = -supplied_mvp[:, 1]
            # cam = MiniCam(orbit.W, orbit.H, orbit.fovy, orbit.fovx, orbit.near, orbit.far,
            #             torch.from_numpy(supplied_mv), torch.from_numpy(supplied_mvp), 1)

        output = renderer.render_from_camera(vertices, faces, cam, background_color=[1, 1, 1])
        rgba_mesh = output['rgba'].squeeze(0)  # (C, W, H)
        print(rgba_mesh.shape)
        image_data = rgba_mesh[:, :, :3].cpu()
        if torch.sum(image_data) >= 3 * 512 * 334:
            print(f"{k} empty image!!")
            continue
        plt.imshow(image_data)
        plt.show()


if __name__ == '__main__':
    main()
