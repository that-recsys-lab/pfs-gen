from .interface import SyntheticDataGenerator
from .torch_model import TorchDataGenerator
from .nce_torch import NCESamplerGenerator
#from .random_walk import RandomWalkGenerator

name_to_class = {
    "fspire": TorchDataGenerator,
    "fspire_nce": NCESamplerGenerator,
#    "random_walk": RandomWalkGenerator,
}
