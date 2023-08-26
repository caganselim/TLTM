from torch import nn
from maskgnn.modeling.obj_encoder.build import OBJ_ENCODER_REGISTRY

@OBJ_ENCODER_REGISTRY.register()
class EncoderMLP(nn.Module):
    """MLP encoder, maps segmentation to latent state."""
    def __init__(self, cfg):
        super(EncoderMLP, self).__init__()

        self.input_dim = cfg.MODEL.MATCHER.OBJ_ENCODER.ENCODER_MLP.INPUT_DIM #28*28
        self.hidden_dim = cfg.MODEL.MATCHER.OBJ_ENCODER.ENCODER_MLP.HIDDEN_DIM
        self.output_dim = cfg.MODEL.MATCHER.OBJ_ENCODER.ENCODER_MLP.OUTPUT_DIM

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim )
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim,  self.output_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, ins):
        h_flat = ins.view(-1, 1, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)