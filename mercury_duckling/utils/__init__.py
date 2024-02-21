from typing import List
from torchvision.transforms.v2 import Compose

from .console import console, console_status, config2table

def get_attribute_name(self, attribute) -> List[str]:
    if isinstance(attribute, Compose):
        return [T.__class__.__name__ for T in attribute.transforms]
    else:
        return [attribute.__class__.__name__]