"""Utility for inference using trained networks"""

import torch

from exercise_2.data.shapenet import ShapeNetVox
from exercise_2.model.cnn3d import ThreeDeeCNN


class InferenceHandler3DCNN:
    """Utility for inference using trained 3DCNN network"""

    def __init__(self, ckpt):
        """
        :param ckpt: checkpoint path to weights of the trained network
        """
        self.model = ThreeDeeCNN(ShapeNetVox.num_classes)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()

    # noinspection PyInterpreter
    def infer_single(self, voxels):
        """
        Infer class of the shape given its voxel grid representation
        :param voxels: voxel grid of shape 32x32x32
        :return: class category name for the voxels, as predicted by the model
        """
        input_tensor = torch.from_numpy(voxels).float().unsqueeze(0).unsqueeze(0)

        ########################################################################
        prediction = self.model(input_tensor).mean(dim=1)
        _, predicted_label = torch.max(prediction, 1)
        class_id = predicted_label.item()
        class_name = ShapeNetVox.classes[class_id]
        ########################################################################

        return class_name
