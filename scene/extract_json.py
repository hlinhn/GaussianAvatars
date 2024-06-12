import os
import sys
import json
import numpy as np
# from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
# from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from typing import NamedTuple, Optional
import math

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: Optional[np.array]
    image_path: str
    image_name: str
    width: int
    height: int
    bg: np.array = np.array([0, 0, 0])
    timestep: Optional[int] = None
    camera_id: Optional[int] = None


def get_image_idx(folder):
    image_path = [x for x in os.listdir(folder)]
    idx = [x[5:-4] for x in image_path]
    return idx


def readManoMeshes(path, mesh_file):
    with open(os.path.join(path, mesh_file)) as json_file:
        contents = json.load(json_file)
        frames = contents['frames']
        mesh_infos = {}
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            frame_idx = frame['idx']
            params = frame['params']
            mesh_infos[frame_idx] = params
    return mesh_infos


def test_one_image(image_path):
    image = Image.open(image_path)
    im_data = np.array(image.convert("RGBA"))
    norm_data = im_data / 255.0
    bg = np.array([1, 1, 1])
    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
    image.save("test.jpg")


def extract_data(source, idx, target):
    saved_data = []
    def convert_param_format(old_format, side='left'):
        new_format = {}
        new_format['betas'] = old_format[side]['shape']
        new_format['transl'] = old_format[side]['trans']
        new_format['global_orient'] = old_format[side]['pose'][:3]
        new_format['hand_pose'] = old_format[side]['pose'][3:]
        return new_format

    with open(source) as json_file:
        contents = json.load(json_file)
        capture_data = contents['0']
        for i, im_idx in enumerate(tqdm(idx)):
            cur_data = {
                "idx": i,
                "params": convert_param_format(capture_data[im_idx]),
                "image_idx": im_idx
            }
            saved_data.append(cur_data)
    saved_format = {"frames": saved_data}
    with open(target, 'w') as writer:
        json.dump(saved_format, writer)
    print(target)


def extract_camera_info(source, sel, target):
    saved_data = []
    with open(source) as json_file:
        contents = json.load(json_file)
        capture_data = contents['0']
        for s in sel:
            odix = s[3:]
            cur_data = {
                "id": s,
                "camrot": capture_data['camrot'][odix],
                "campos": capture_data['campos'][odix],
                "focal": capture_data['focal'][odix],
                "princpt": capture_data['princpt'][odix]
            }
            saved_data.append(cur_data)
    saved_format = {"cameras": saved_data}
    with open(target, 'w') as writer:
        json.dump(saved_format, writer)
    print(target)


def readManoCameras(path, mesh_file, camera_file, white_background=False):
    source_folder = "/home/halinh/external_data/InterHand2.6M_30fps_batch1/images/test/Capture0/ROM03_LT_No_Occlusion/"
    idx_to_image = {}
    camera_infos = {}
    with open(os.path.join(path, mesh_file)) as json_file:
        contents = json.load(json_file)
        frames = contents['frames']
        for frame in frames:
            idx_to_image[frame['idx']] = frame['image_idx']
    with open(os.path.join(path, camera_file)) as camera_json:
        contents = json.load(camera_json)
        cameras = contents['cameras']
        for c in cameras:
            R = np.array(c['camrot'])
            T = np.array(c['campos'])
            focal = np.array(c['focal'])
            cam_id = c['id']
            camera_infos[cam_id] = {'R': R, 'T': T, 'focal': focal}

    cam_infos = []
    total_idx = 0
    for c in camera_infos.keys():
        for m in idx_to_image.keys():
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            image_path = os.path.join(source_folder, c, f"image{idx_to_image[m]}.jpg")
            image_name = Path(image_path).stem
            image = Image.open(image_path)
            # we need the background masking for this one
            im_data = np.array(image.convert("RGBA"))
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            width, height = image.size
            focal_c = np.array(camera_infos[c]['focal'])
            fovy = focal2fov(focal_c[1], height)
            fovx = focal2fov(focal_c[0], width)
            cam_infos.append(CameraInfo(
                uid=total_idx, R=camera_infos[c]['R'], T=camera_infos[c]['T'], 
                FovY=fovy, FovX=fovx, bg=bg, image=image, 
                image_path=image_path, image_name=image_name, 
                width=width, height=height,
                timestep=m, camera_id=c))
            total_idx += 1
    return cam_infos


def split_views(source_folder):
    views = [x for x in os.listdir(source_folder)]
    train_sel = []
    test_sel = []
    val_sel = []
    for v in views:
        roll = np.random.rand()
        if roll < 0.2:
            train_sel.append(v)
        elif roll < 0.3:
            test_sel.append(v)
        elif roll < 0.4:
            val_sel.append(v)
    return train_sel, val_sel, test_sel


def convert_param_format(old_format, side='left'):
    new_format = {}
    new_format['betas'] = old_format[side]['shape']
    new_format['transl'] = old_format[side]['trans']
    new_format['global_orient'] = old_format[side]['pose'][:3]
    new_format['hand_pose'] = old_format[side]['pose'][3:]
    return new_format


def read_mano_param(filepath):
    param = np.load(filepath)
    for k in param.keys():
        print(f"{k}: {param[k].shape} :{param[k].dtype}")


def main():
    # test_one_image(sys.argv[1])
    source_folder = "/mnt/data/InterHand/InterHand2.6M_30fps_batch1"
    train_image = "images/test/Capture0/ROM03_LT_No_Occlusion"
    sample_cam = "cam400262"
    source_mano_annot = "annotations/test/InterHand2.6M_test_MANO_NeuralAnnot.json"
    camera_annot = "annotations/test/InterHand2.6M_test_camera.json"
    idx = get_image_idx(os.path.join(source_folder, train_image, sample_cam))
    # print(len(idx))
    idx = sorted(idx)
    print(len(idx[::30]))
    source_file = os.path.join(source_folder, source_mano_annot)
    saved_data = []
    target = "mano_frames_3fps.json"

    with open(source_file) as f:
        contents = json.load(f)
        print(len(contents))
        capture_data = contents['0']
        for i, im_idx in enumerate(tqdm(idx[::10])):
            cur_data = {
                "idx": i,
                "params": convert_param_format(capture_data[im_idx]),
                "image_idx": im_idx
            }
            saved_data.append(cur_data)
    if False:
        test_mano_param = {'betas': np.array(saved_data[0]['params']['betas']).astype(np.float32)}
        for k in ['transl', 'global_orient', 'hand_pose']:
            accum = []
            for data in saved_data:
                accum.append(data['params'][k])
            accum = np.array(accum).astype(np.float32)
            test_mano_param[k] = accum
        np.savez("mano_param_test.npz", **test_mano_param)
        read_mano_param("mano_param_test.npz")
    saved_format = {"frames": saved_data}
    with open(target, 'w') as writer:
        json.dump(saved_format, writer)
    print(target)
    # extract_data(os.path.join(source_folder, source_mano_annot), idx[::30], "mano_frames_test.json")
    # readManoMeshes(".", "mano_frames.json")
    # train_v, val_v, test_v = split_views(os.path.join(source_folder, train_image))
    # extract_camera_info(os.path.join(source_folder, camera_annot), train_v, "train_cam_try.json")
    # extract_camera_info(os.path.join(source_folder, camera_annot), val_v, "val_cam_try.json")
    # extract_camera_info(os.path.join(source_folder, camera_annot), test_v, "test_cam_try.json")
    # lst = readManoCameras(".", "mano_frames_test.json", "test_cam_try.json")
    # print(len(lst))


if __name__ == '__main__':
    # read_mano_param(sys.argv[1])
    main()