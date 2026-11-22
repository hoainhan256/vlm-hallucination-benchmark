"""Abstract benchmark interface preserving benchmark-specific protocols."""

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping


class BaseBenchmark(ABC):
    @abstractmethod
    def read_dataset(self) -> Iterable[Mapping[str, Any]]:
        """Read the configured dataset without downloading or mutating it."""

    @abstractmethod
    def normalize_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        """Normalize one record into the benchmark-owned schema."""

    @abstractmethod
    def prepare_inputs(self, record: Mapping[str, Any]) -> Any:
        """Prepare model input using the verified original protocol."""

    @abstractmethod
    def call_model(self, model: Any, prepared_input: Any) -> Any:
        """Call the model without changing benchmark-specific prompt rules."""

    @abstractmethod
    def score(self, predictions: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Apply the verified original benchmark scoring protocol."""

    @abstractmethod
    def export_results(self, predictions: Iterable[Mapping[str, Any]], output_dir: str) -> None:
        """Export run artifacts according to the result schema."""
