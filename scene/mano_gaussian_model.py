# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
# from vht.model.flame import FlameHead
from flame_model.mano import MANO

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes


class ManoGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, is_rhand=False):
        super().__init__(sh_degree)

        self.mano_model = MANO(
            is_rhand=is_rhand
        ).cuda()
        if not is_rhand:
            self.mano_model.shapedirs[:,0,:] *= -1
        self.mano_param = None
        self.mano_param_orig = None

        example_mano = self.mano_model(
            torch.zeros([1, 10]).cuda(),
            torch.zeros([1, 3]).cuda(),
            torch.zeros([1, 45]).cuda(),
            torch.zeros([1, 3]).cuda())
        example_mano_mesh = Meshes(example_mano.vertices, self.mano_model.faces_tensor.repeat(1, 1, 1))
        self.mesh_divider = SubdivideMeshes(example_mano_mesh)
        # print(self.mano_model.faces_tensor.shape)
        # print(self.mesh_divider._subdivided_faces.shape)
        # self.mesh_divider = self.mano_model.faces_tensor
        # binding is initialized once the mesh topology is known
        number_of_faces = len(self.mesh_divider._subdivided_faces)
        if self.binding is None:
            self.binding = torch.arange(number_of_faces).cuda()
            self.binding_counter = torch.ones(number_of_faces, dtype=torch.int32).cuda()

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.mano_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            # num_verts = self.mano_model.v_template.shape[0]

            T = self.num_timesteps

            self.mano_param = {
                'betas': torch.from_numpy(meshes[0]['betas']),
                'global_orient': torch.zeros([T, meshes[0]['global_orient'].shape[0]]),
                'hand_pose': torch.zeros([T, meshes[0]['hand_pose'].shape[0]]),
                'transl': torch.zeros([T, 3])
            }

            for i, mesh in pose_meshes.items():
                self.mano_param['global_orient'][i] = torch.from_numpy(mesh['global_orient'])
                self.mano_param['hand_pose'][i] = torch.from_numpy(mesh['hand_pose'])
                self.mano_param['transl'][i] = torch.from_numpy(mesh['transl'])
            
            for k, v in self.mano_param.items():
                self.mano_param[k] = v.float().cuda()
            
            self.mano_param_orig = {k: v.clone() for k, v in self.mano_param.items()}
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def update_mesh_by_param_dict(self, mano_param):
        if 'betas' in mano_param:
            betas = mano_param['betas']
        else:
            betas = self.mano_param['betas']

        mano_output = self.mano_model(
            betas[None, ...],
            mano_param['global_orient'].cuda(),
            mano_param['hand_pose'].cuda(),
            mano_param['transl'].cuda()
        )
        subdivided_mesh = self.mesh_divider(Meshes(mano_output.vertices, self.mano_model.faces_tensor.repeat(1, 1, 1)))
        # verts = mano_output.vertices
        # print(mano_output.vertices.shape)
        verts = subdivided_mesh.verts_padded()
        # print(verts.shape)
        verts_cano = mano_output.v_shaped
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        mano_param = self.mano_param_orig if original and self.mano_param_orig != None else self.mano_param

        mano_output = self.mano_model(
            mano_param['betas'][None, ...],
            mano_param['global_orient'][[timestep]].cuda(),
            mano_param['hand_pose'][[timestep]].cuda(),
            mano_param['transl'][[timestep]].cuda()
        )
        subdivided_mesh = self.mesh_divider(Meshes(mano_output.vertices, self.mano_model.faces_tensor.repeat(1, 1, 1)))
        verts = subdivided_mesh.verts_padded() # mano_output.vertices
        # verts = mano_output.vertices
        verts_cano = mano_output.v_shaped
        self.update_mesh_properties(verts, verts_cano)
    
    def update_mesh_properties(self, verts, verts_cano):
        # need to check the dimension here
        faces = self.mesh_divider._subdivided_faces # self.mano_model.faces_tensor
        triangles = verts[:, faces]
        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces, return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization, not using yet
        self.verts_cano = verts_cano
    
    def compute_dynamic_offset_loss(self):
        return 0
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        return 0
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        # if self.not_finetune_flame_params:
        #     return

        # # shape
        # self.flame_param['shape'].requires_grad = True
        # param_shape = {'params': [self.flame_param['shape']], 'lr': 1e-5, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # pose
        train_param = True
        self.mano_param['global_orient'].requires_grad = train_param
        self.mano_param['hand_pose'].requires_grad = train_param
        params = [
            self.mano_param['global_orient'],
            self.mano_param['hand_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # translation
        self.mano_param['transl'].requires_grad = train_param
        param_trans = {'params': [self.mano_param['transl']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # betas
        self.mano_param['betas'].requires_grad = train_param
        param_expr = {'params': [self.mano_param['betas']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)

        # # static_offset
        # self.flame_param['static_offset'].requires_grad = True
        # param_static_offset = {'params': [self.flame_param['static_offset']], 'lr': 1e-6, "name": "static_offset"}
        # self.optimizer.add_param_group(param_static_offset)

        # # dynamic_offset
        # self.flame_param['dynamic_offset'].requires_grad = True
        # param_dynamic_offset = {'params': [self.flame_param['dynamic_offset']], 'lr': 1.6e-6, "name": "dynamic_offset"}
        # self.optimizer.add_param_group(param_dynamic_offset)

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "mano_param.npz"
        mano_param = {k: v.cpu().numpy() for k, v in self.mano_param.items()}
        np.savez(str(npz_path), **mano_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "mano_param.npz"
            mano_param = np.load(str(npz_path))
            mano_param = {k: torch.from_numpy(v).cuda() for k, v in mano_param.items()}

            self.mano_param = mano_param
            self.num_timesteps = self.mano_param['betas'].shape[0]  # required by viewers
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            mano_param = np.load(str(motion_path))
            mano_param = {k: torch.from_numpy(v).cuda() for k, v in mano_param.items() if v.dtype == np.float32}

            self.mano_param['transl'] = mano_param['transl']
            self.mano_param['global_orient'] = mano_param['global_orient']
            self.mano_param['hand_pose'] = mano_param['hand_pose']
            self.mano_param['betas'] = mano_param['betas']
            self.num_timesteps = self.mano_param['betas'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
