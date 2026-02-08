"""continualcode — self-improving coding agent with online SDPO."""

__version__ = "0.5.0"

from .cli import Config
from .train import SDPOConfig, ContinualSDPOSession, SampledCompletion

__all__ = ["Config", "SDPOConfig", "ContinualSDPOSession", "SampledCompletion", "__version__"]
