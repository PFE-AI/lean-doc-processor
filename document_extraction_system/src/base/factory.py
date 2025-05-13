from typing import Dict, Type
from ..models import DocumentType
from .extractor import BaseDocumentExtractor


class DocumentExtractorFactory:
    """Factory for creating document extractors"""
    
    _extractors: Dict[DocumentType, Type[BaseDocumentExtractor]] = {}
    
    @classmethod
    def register_extractor(cls, 
                          document_type: DocumentType, 
                          extractor_class: Type[BaseDocumentExtractor]):
        """Register a new extractor"""
        cls._extractors[document_type] = extractor_class
    
    @classmethod
    def create_extractor(cls, 
                        document_type: DocumentType,
                        api_key: Optional[str] = None) -> BaseDocumentExtractor:
        """Create an extractor for the given document type"""
        if document_type not in cls._extractors:
            raise ValueError(f"No extractor registered for document type: {document_type}")
        
        extractor_class = cls._extractors[document_type]
        return extractor_class(api_key=api_key)