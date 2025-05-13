from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Union
from pathlib import Path
import time
import logging

from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai.chat_models import ChatMistralAI

from .models import DocumentType, ExtractedData


logger = logging.getLogger(__name__)


class BaseDocumentExtractor(ABC):
    """Abstract base class for document extractors"""
    
    def __init__(self, 
                 document_type: DocumentType,
                 api_key: Optional[str] = None,
                 model_name: str = "mistral-small-latest",
                 temperature: float = 0.1):
        self.document_type = document_type
        self.llm = ChatMistralAI(
            model_name=model_name,
            mistral_api_key=api_key,
            temperature=temperature,
            max_tokens=2048
        )
        
    @abstractmethod
    def create_extraction_prompt(self) -> PromptTemplate:
        """Create the prompt template for extraction"""
        pass
    
    @abstractmethod
    def create_output_model(self) -> Type[BaseModel]:
        """Create the Pydantic model for structured output"""
        pass
    
    @abstractmethod
    def validate_extracted_data(self, data: Dict) -> List[str]:
        """Validate the extracted data"""
        pass
    
    def extract_from_text(self, 
                         text: str,
                         use_structured_output: bool = True) -> ExtractedData:
        """Extract information from text"""
        start_time = time.time()
        
        try:
            if use_structured_output:
                # Use structured output with Pydantic
                output_model = self.create_output_model()
                parser = PydanticOutputParser(pydantic_object=output_model)
                
                prompt = self.create_extraction_prompt()
                prompt = prompt.partial(format_instructions=parser.get_format_instructions())
                
                chain = prompt | self.llm | parser
                result = chain.invoke({"text": text})
                extracted_fields = result.dict()
            else:
                # Use unstructured output
                prompt = self.create_extraction_prompt()
                chain = prompt | self.llm | StrOutputParser()
                result = chain.invoke({"text": text})
                extracted_fields = self._parse_unstructured_output(result)
            
            # Validate extracted data
            validation_errors = self.validate_extracted_data(extracted_fields)
            
            processing_time = time.time() - start_time
            
            return ExtractedData(
                document_type=self.document_type,
                raw_text=text,
                extracted_fields=extracted_fields,
                validation_errors=validation_errors,
                processing_time=processing_time,
                metadata={
                    "extraction_method": "structured" if use_structured_output else "unstructured",
                    "model_name": self.llm.model_name
                }
            )
            
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            return ExtractedData(
                document_type=self.document_type,
                raw_text=text,
                extracted_fields={},
                validation_errors=[f"Extraction error: {str(e)}"],
                processing_time=time.time() - start_time
            )
    
    def _parse_unstructured_output(self, output: str) -> Dict:
        """Parse unstructured output to dictionary"""
        import json
        try:
            # Try to parse as JSON first
            return json.loads(output)
        except:
            # Fallback to simple key-value parsing
            lines = output.strip().split('\n')
            result = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    result[key.strip()] = value.strip()
            return result