from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.raw_document import RawDocument

class BaseExtractor(ABC):
    # base interface for all document extractors

    @abstractmethod
    def extract(self,pdf_path:Path) -> RawDocument:
        # extract a pdf into out internal rawdocument representation

        raise NotImplementedError(
            "Docling extractor implementation will be added after "
            "the playground api is verified"
        )