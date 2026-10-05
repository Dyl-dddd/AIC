from types import SimpleNamespace

import numpy as np
import torch

from steel_defect.inference import InferenceOptions, predict_candidates_array


class FakeBoxes(SimpleNamespace):
    def __len__(self):
        return len(self.conf)


class CapturingModel:
    def __init__(self):
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(
                boxes=FakeBoxes(
                    xyxy=torch.tensor([[10.0, 20.0, 40.0, 50.0]]),
                    conf=torch.tensor([0.9]),
                    cls=torch.tensor([0.0]),
                )
            )
            for _ in source
        ]


def test_tta_flag_is_forwarded_to_model_predict():
    model = CapturingModel()
    image = np.zeros((100, 100), dtype=np.uint8)
    rows, image_size = predict_candidates_array(
        model,
        image,
        InferenceOptions(tile_size=100, batch=1, tta=True),
    )
    assert image_size == (100, 100)
    assert len(rows) == 1
    assert len(model.calls) == 1
    assert model.calls[0]["augment"] is True
