"""Base collector interface for all source collectors."""

import logging
from abc import ABC, abstractmethod

from app.factory.models import RawItem

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base class for source collectors."""

    name: str = "unknown"

    @abstractmethod
    def collect(self) -> list[RawItem]:
        """Fetch new items from the source. Returns list of RawItem."""
        ...

    def run(self) -> list[RawItem]:
        """Collect with error handling and logging."""
        logger.info("[%s] Starting collection...", self.name)
        try:
            items = self.collect()
            logger.info("[%s] Collected %d items", self.name, len(items))
            return items
        except Exception as e:
            logger.error("[%s] Collection failed: %s", self.name, e, exc_info=True)
            return []
