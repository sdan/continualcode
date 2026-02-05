"""continualcode — self-improving coding agent with online SDPO."""

__version__ = "0.5.0"

from .cli import Config
from .train import SDPOConfig, ContinualSDPOSession

__all__ = ["Config", "SDPOConfig", "ContinualSDPOSession", "__version__"]
