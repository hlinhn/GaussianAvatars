from scene import Scene, ManoGaussianModel
from mesh_renderer import NVDiffRenderer
from scene.cameras import MiniCam, Camera
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageFilter
import nvdiffrast.torch as dr
import os
import json
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights,
    DirectionalLights,
    PerspectiveCameras,
    Materials,
    SoftPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRendererWithFragments,
    MeshRasterizer,
    TexturesVertex)
from tqdm import tqdm
import sys
from argparse import ArgumentParser


def torch_render_mesh(mesh, face, cam_param, render_shape, hand_type):
    t, R = np.array(cam_param['campos'], dtype=np.float32).reshape(3), np.array(cam_param['camrot'], dtype=np.float32).reshape(3,3)
    t = -np.dot(R,t.reshape(3,1)).reshape(3) / 1000 # -Rt -> t
    mesh = np.dot(R, mesh.cpu().transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    mesh = torch.from_numpy(mesh).float().cuda()[None, :, :]
    face = torch.from_numpy(face.astype(np.int32)).cuda()[None,:,:]
    batch_size, vertex_num = mesh.shape[:2]
    # mesh = mesh / 1000 # milimeter to meter

    textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cuda())
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    cameras = PerspectiveCameras(focal_length=torch.FloatTensor(cam_param['focal']).cuda()[None, :],
                                principal_point=torch.FloatTensor(cam_param['princpt']).cuda()[None, :],
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    if hand_type == 'right':
        color = ((1.0, 0.0, 0.0),)
    else:
        color = ((0.0, 1.0, 0.0),)
    materials = Materials(device='cuda', specular_color=color, shininess=0)

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
        images = images[:,:,:,:3] #* 255
        depthmaps = fragments.zbuf

    return images, depthmaps
   

def manoProjectionMatrix(znear, zfar, focal, center, size):
    proj_mat = np.array(
        [
            [2 * focal[0] / size[0], 0, (size[0] - 2 * center[0]) / size[0], 0],
            [0, 2 * focal[1] / size[1], -(size[1] - 2 * center[1]) / size[1], 0],
            [
                0,
                0,
                -(zfar + znear) / (zfar - znear),
                -(2 * zfar * znear) / (zfar - znear),
            ],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )
    return proj_mat


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
            focal = np.copy(np.array(c['focal']))
            fovx = focal2fov(focal[0], w)
            fovy = focal2fov(focal[1], h)
            # proj = make_perspective(fovy, fovx, near, far)
            proj = manoProjectionMatrix(near, far, focal, c['princpt'], (w, h))
            transform_mat = np.eye(4)
            transform_mat[:3, :3] = np.copy(np.array(c['camrot']))
            R = transform_mat[:3, :3]
            transform_mat[:3, 3] = np.matmul(-R, np.copy(np.array(c['campos']))) * 0.001
            transform_mat[1, :] = -transform_mat[1, :]
            transform_mat[2, :] = -transform_mat[2, :]
            R = transform_mat[:3, :3].transpose()
            T = transform_mat[:3, 3]
            a_mvp = np.matmul(proj, transform_mat).astype(np.float32)
            supplied_mv = transform_mat
            supplied_mv = supplied_mv.T
            supplied_mv[:, 1] = -supplied_mv[:, 1]
            supplied_mv[:, 2] = -supplied_mv[:, 2]
            supplied_mvp = a_mvp
            supplied_mvp = supplied_mvp.T
            supplied_mvp[:, 1] = -supplied_mvp[:, 1]

            cam_id = c['id']
            camera_infos[cam_id] = {'R': R, 'T': T, 'focal': focal,
                                    'w': w, 'h': h, 'fovx': fovx, 'fovy': fovy,
                                    'mv': supplied_mv, 'mvp': supplied_mvp,
                                    'near': near, 'far': far, 'princpt': c['princpt'],
                                    'camrot': c['camrot'], 'campos': c['campos']}
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

# After the shapedirs fix, there is still an offset -> the projection matrix has to be 
# the same as the one in the pytorch3d version


def readManoMeshes(mesh_file):
    with open(mesh_file) as json_file:
        contents = json.load(json_file)
        frames = contents['frames']
        mesh_infos = {}
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            frame_idx = frame['image_idx']
            params = frame['params']
            params_np = {}
            for k, v in params.items():
                params_np[k] = torch.from_numpy(np.asarray(v)).float().to('cuda')
            mesh_infos[frame_idx] = params_np
    return mesh_infos


def convert_point_cloud(model_path):
    import open3d as o3d
    gaussians = ManoGaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(model_path,
                            "point_cloud",
                            "iteration_" + str(120000),
                            "point_cloud.ply"),
                       has_target=False)
    actual_cloud = gaussians.get_xyz.cpu().detach().numpy()
    print(actual_cloud.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(actual_cloud)
    o3d.io.write_point_cloud(os.path.join(model_path, "actual_cloud.ply"), pcd, write_ascii=True)


def semantic_segmentation(image_list, camera_list, sequence, k):
    root_folder = "/mnt/data/InterHand/InterHand2.6M_30fps_batch1"
    image_folder = f"test/Capture0/{sequence}"
    target_path = os.path.join(root_folder, "semantic_masks", image_folder)
    os.makedirs(target_path, exist_ok=True)
    image_path = os.path.join(root_folder, "images", image_folder)
    model = torch.hub.load(
        repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
        model='hand_segmentor', 
        pretrained=True
    )

    model.eval()    
    mesh_info = readManoMeshes(image_list)
    cam_info = read_example_camera(camera_list)
    test_cam = cam_info[k]
    for im_name, mesh in mesh_info.items():
        target_mask_path = os.path.join(target_path, k, f'image{im_name}.jpg')
        if os.path.isfile(target_mask_path):
            continue


def make_masks(image_list, camera_list, sequence, k):
    root_folder = "/mnt/data/InterHand/InterHand2.6M_30fps_batch1"
    image_folder = f"test/Capture0/{sequence}"
    target_path = os.path.join(root_folder, "masks", image_folder)
    os.makedirs(target_path, exist_ok=True)
    image_path = os.path.join(root_folder, "images", image_folder)
    mesh_info = readManoMeshes(image_list)
    cam_info = read_example_camera(camera_list)
    test_cam = cam_info[k]
    hand_model = ManoGaussianModel(sh_degree=1)
    renderer = NVDiffRenderer(use_opengl=False)
    faces = torch.from_numpy(hand_model.mano_model.faces.astype(np.int32))
    for im_name, mesh in mesh_info.items():
        target_mask_path = os.path.join(target_path, k, f'image{im_name}.jpg')
        if os.path.isfile(target_mask_path):
            continue
        hand_output = hand_model.mano_model(
            mesh['betas'][None],
            mesh['global_orient'][None],
            mesh['hand_pose'][None],
            mesh['transl'][None]
        )
        vertices = hand_output.vertices
        faces = faces.to(vertices.device)

        os.makedirs(os.path.join(target_path, k), exist_ok=True)
        cam = MiniCam(test_cam['w'], test_cam['h'], test_cam['fovy'], test_cam['fovx'],
                        test_cam['near'], test_cam['far'],
                        torch.from_numpy(np.array(test_cam['mv'])),
                        torch.from_numpy(np.array(test_cam['mvp'])), 1)
        output = renderer.render_from_camera(vertices, faces, cam, background_color=[0, 0, 0])
        rgba_mesh = output['rgba'].squeeze(0)  # (C, W, H)
        diffuse = (rgba_mesh.cpu().numpy()[:, :, :3] > 0).astype(np.float32)
        diffuse_im = Image.fromarray((diffuse * 255).astype(np.uint8))
        dilation_img = diffuse_im.filter(ImageFilter.MaxFilter(5))
        dilation_img.save(target_mask_path)


def main():
    # check 422
    hand_model = ManoGaussianModel(sh_degree=1)
    print(hand_model.mano_model.faces.shape)
    # npz_path = "output/59b3d072-1/point_cloud/iteration_2/mano_param.npz"
    # npz_path = "output/b6e05868-e/point_cloud/iteration_1/mano_param.npz"
    npz_path = "output/f335a4be-6/point_cloud/iteration_100/mano_param.npz"
    folder = "/mnt/data/InterHand/InterHand2.6M_30fps_batch1/images/test/Capture0/ROM03_LT_No_Occlusion"
    mano_param = np.load(str(npz_path))
    mano_param = {k: torch.from_numpy(v).cuda() for k, v in mano_param.items()}
    # hand_model.mano_model.shapedirs[:,0,:] *= -1
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
    _, axs = plt.subplots(len(infos) // 3 + 1, 3, figsize=(15, 15))
    for i, k in enumerate(infos.keys()):
        # cam_info_to_minicam(v)
        # exit(0)
        print(f"{k}")
        test_cam = infos[k]
        cam = MiniCam(test_cam['w'], test_cam['h'], test_cam['fovy'], test_cam['fovx'],
                      test_cam['near'], test_cam['far'],
                      torch.from_numpy(np.array(test_cam['mv'])),
                      torch.from_numpy(np.array(test_cam['mvp'])), 1)
        if True:
            output = renderer.render_from_camera(vertices, faces, cam, background_color=[0, 0, 0])
            rgba_mesh = output['rgba'].squeeze(0)  # (C, W, H)

            # diffuse = output['diffuse'].squeeze(0)
            diffuse = (rgba_mesh.cpu().numpy()[:, :, :3] > 0).astype(np.float32)
            diffuse_im = Image.fromarray((diffuse * 255).astype(np.uint8))
            dilation_img = diffuse_im.filter(ImageFilter.MaxFilter(5))
            # print(rgba_mesh.shape)
            # image_data = dilation_img[:, :, :3]
            # if torch.sum(image_data) >= 3 * 512 * 334:
            #     print(f"{k} empty image!!")
            #     continue
        else:
            output, _ = torch_render_mesh(vertices[0], faces.cpu().numpy(), infos[k], (512, 334), 'left')
            rgb_mesh = output[0]
            image_data = rgb_mesh.cpu()
        row = i // 3
        col = (i % 3) 
        orig_im = Image.open(os.path.join(folder, k, "image15350.jpg"))
        # image_data = (np.array(orig_im) * 0.5 + image_data.numpy() * 0.5 * 255).astype(np.int32)
        # image_data = image_data.numpy()        
        image_data = np.array(dilation_img)
        image_data = (image_data > 0).astype(np.int32)
        image_data = np.multiply(image_data, np.array(orig_im)).astype(np.int32)
        axs[row][col].imshow(image_data)
        axs[row][col].set_title(k)
    plt.savefig("visualize.png")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--action', type=str, default='mask')
    parser.add_argument('--sequence', type=str, default='ROM03_LT_No_Occlusion')
    parser.add_argument('--target', type=str, default='test_cam_20.json')
    parser.add_argument('--image_list', type=str, default='mano_frames_1fps.json')
    parser.add_argument('--cam', type=str, default='')
    args = parser.parse_args()

    if args.action == 'mask':
        make_masks(args.image_list, args.target, args.sequence, args.cam)
