from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("exercise_3/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        #################################################################
        # Apply truncation to sdf and df
        input_sdf = np.clip(input_sdf, a_min=-3, a_max=3)
        target_df = np.clip(target_df, a_min=0, a_max=3)

        # Stack (distances, sdf sign) for the input sdf
        sdf_sign = np.sign(input_sdf)
        input_sdf = np.stack([np.abs(input_sdf), sdf_sign], axis=0)

        # Log-scale target df
        target_df = np.log(target_df + 1)
        #################################################################

        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        #################################################################
        # Add code to move batch to device
        batch['input_sdf'] = batch['input_sdf'].to(device)
        batch['target_df'] = batch['target_df'].to(device)
        #################################################################

    @staticmethod
    def get_shape_sdf(shapenet_id):
        ########################################################################
        # Implement sdf data loading
        category_id, shape_id_with_trajectory = shapenet_id.split('/')
        path = ShapeNet.dataset_sdf_path / category_id / f"{shape_id_with_trajectory}.sdf"

        dim = np.fromfile(path, dtype=np.uint64, count=3)
        data_num = (dim[0] * dim[1] * dim[2])

        byte_num = 8  # uint64 has 8 bytes
        sdf = np.fromfile(path, dtype=np.float32, offset=(byte_num * 3), count=data_num)

        sdf_reshaped = sdf.reshape(sdf.shape[0], 1)
        sdf = sdf_reshaped.reshape(dim[0], dim[1], dim[2])
        return sdf
        ########################################################################

    @staticmethod
    def get_shape_df(shapenet_id):
        ########################################################################
        # Implement df data loading
        category_id, shape_id_with_trajectory = shapenet_id.split('/')
        path = ShapeNet.dataset_df_path / category_id / f"{shape_id_with_trajectory}.df"

        dim = np.fromfile(path, dtype=np.uint64)
        data_num = (dim[0] * dim[1] * dim[2])

        byte_num = 8  # uint64 has 8 bytes
        df = np.fromfile(path, dtype=np.float32, offset=(byte_num * 3), count=data_num)

        df_reshaped = df.reshape(df.shape[0], 1)
        df = df_reshaped.reshape(dim[0], dim[1], dim[2])
        return df
        ########################################################################
