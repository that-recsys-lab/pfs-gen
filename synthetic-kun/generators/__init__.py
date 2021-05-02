from .interface import SyntheticDataGenerator
from .torch_model import TorchDataGenerator
from .random_walk import RandomWalkGenerator

name_to_class = {
    "fspire_mixed": TorchDataGenerator,
    "random_walk": RandomWalkGenerator,
}
