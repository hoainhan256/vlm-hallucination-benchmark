"""Abstract interface for VLMs. No model weights are loaded by this module."""

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping


class BaseVLM(ABC):
    @classmethod
    @abstractmethod
    def load_config(cls, model_config: Mapping[str, Any]) -> "BaseVLM":
        """Validate configuration and construct a model adapter."""

    @abstractmethod
    def initialize_processor(self) -> None:
        """Initialize a processor in a later implementation phase."""

    @abstractmethod
    def load_model(self) -> None:
        """Load approved official weights in a later implementation phase."""

    @abstractmethod
    def infer(self, inputs: Iterable[Any], generation_config: Mapping[str, Any]) -> Iterable[Any]:
        """Run inference. It must not be implemented as a placeholder result."""

    @abstractmethod
    def release_resources(self) -> None:
        """Release accelerator and CPU resources."""
