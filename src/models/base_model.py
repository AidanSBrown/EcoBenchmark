from abc import ABC, abstractmethod
from typing import List, Dict, Any

class TreeDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str, config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass