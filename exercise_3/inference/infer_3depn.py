import numpy as np
import torch
from skimage.measure import marching_cubes

from exercise_3.model.threedepn import ThreeDEPN


class InferenceHandler3DEPN:
    def __init__(self, ckpt):
        """
        :param ckpt: checkpoint path to weights of the trained network
        """
        self.model = ThreeDEPN()
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()
        self.truncation_distance = 3

    def infer_single(self, input_sdf, target_df):
        """
        Reconstruct a full shape given a partial observation
        :param input_sdf: Input grid with partial SDF of shape 32x32x32
        :param target_df: Target grid with complete DF of shape 32x32x32
        :return: Tuple with mesh representations of input, reconstruction, and target
        """

        ###################################################################################
        # Apply truncation distance: SDF values should lie within -3 and 3, DF values between 0 and 3
        input_sdf = np.clip(input_sdf, a_min=-1 * self.truncation_distance, a_max=self.truncation_distance)
        target_df = np.clip(target_df, a_min=0, a_max=self.truncation_distance)
        ###################################################################################

        # Stack (distances, sdf sign) for the input sdf
        sdf_sign = np.sign(input_sdf)
        input_sdf_stacked = np.stack([np.abs(input_sdf), sdf_sign], axis=0)

        # Log-scale target df
        target_df = np.log(target_df + 1)

        with torch.no_grad():
            ###################################################################################
            # Pass input in the right format though the network and
            # revert the log scaling by applying exp and subtracting 1
            input_tensor = torch.from_numpy(input_sdf_stacked).float().unsqueeze(0)
            reconstructed_df = self.model(input_tensor)      # output was log-scaled
            reconstructed_df = np.exp(reconstructed_df) - 1  # invert log-scaling
            ###################################################################################

        input_sdf = np.abs(input_sdf)
        input_mesh = marching_cubes(input_sdf, level=1)
        reconstructed_mesh = marching_cubes(reconstructed_df.squeeze(0).numpy(), level=1)
        target_mesh = marching_cubes(target_df, level=1)
        return input_mesh, reconstructed_mesh, target_mesh
