from .dae import DenoisingAutoencoder as DAE
from .sdae import StackedDenoisingAutoEncoder as SDAE
# from .sdae_mlp_cnn_half import StackedDenoisingAutoEncoder as SDAE
from .dec import DeepEmbeddingCluster as DEC
from .ddc import DeepDivergenceCluster as DDC
from .ddc_resnet import DeepDivergenceCluster as DDC_resnet
from .image_feature_extract import ImageProcess

Models = dict(DAE=DAE,
              SDAE=DAE,
              DDC=DDC,
              DEC=DEC,
              DDC_resnet=DDC_resnet)