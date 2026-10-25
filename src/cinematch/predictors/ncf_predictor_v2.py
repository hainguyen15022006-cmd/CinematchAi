from typing import List, Union
import numpy as np
import torch
import torch.nn as nn


class NCFPredictorV2:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def predict(
        self,
        user_ids: Union[List[int], np.ndarray, torch.Tensor],
        item_ids: Union[List[int], np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        if not isinstance(user_ids, torch.Tensor):
            user_ids_tensor = torch.tensor(user_ids, dtype=torch.long)
        else:
            user_ids_tensor = user_ids.long()

        if not isinstance(item_ids, torch.Tensor):
            item_ids_tensor = torch.tensor(item_ids, dtype=torch.long)
        else:
            item_ids_tensor = item_ids.long()

        user_ids_tensor = user_ids_tensor.to(self.device)
        item_ids_tensor = item_ids_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(user_ids_tensor, item_ids_tensor)
            scores = outputs.squeeze().cpu().numpy()

        if scores.ndim == 0:
            scores = np.array([scores.item()])

        return scores