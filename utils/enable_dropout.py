import torch

def enable_dropout_test(model, p_rate: float = 0.10):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p_rate
            m.train()
    return model
