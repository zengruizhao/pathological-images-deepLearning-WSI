"""
ouput the model pdf
"""
import torch
from torchviz import make_dot
from torchvision.models import AlexNet
import sys
sys.path.append('..')
from src.model import SEResNext50

model = SEResNext50()

x = torch.randn(1, 3, 144, 144).requires_grad_(True)
y = model(x)
vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
vis_graph.view()
