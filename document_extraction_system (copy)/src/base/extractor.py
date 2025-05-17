from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Union
from pathlib import Path
import time
import logging
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_mistralai.chat_models import ChatMistralAI


# Custom exception for rate limiting since langchain_core doesn't have it
class RateLimitError(Exception):
    """Exception raised when API rate limits are exceeded"""
    pass

from .models import DocumentType, ExtractedData


logger = logging.getLogger(__name__)


class BaseDocumentExtractor(ABC):
    """Abstract base class for document extractors with rate limit handling"""
    
    def __init__(self, 
                 document_type: DocumentType,
                 api_key: Optional[str] = None,
                 model: str = "mistral-small-latest",
                 temperature: float = 0.1,
                 max_retries: int = 5,
                 initial_retry_delay: float = 1.0,
                 max_retry_delay: float = 60.0,
                 jitter_factor: float = 0.25):
        """
        Initialize the document extractor with rate limiting parameters.
        
        Args:
            document_type: Type of document to extract
            api_key: API key for Mistral AI
            model: Model name to use
            temperature: Temperature for generation
            max_retries: Maximum number of retries on rate limit errors
            initial_retry_delay: Initial delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds
            jitter_factor: Random jitter factor to add to delay (0.0-1.0)
        """
        self.document_type = document_type
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.jitter_factor = jitter_factor
        
        self.llm = ChatMistralAI(
            model=model,
            api_key=api_key,
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
    
    def _add_jitter(self, delay: float) -> float:
        """Add random jitter to delay to prevent synchronized retries"""
        jitter = delay * self.jitter_factor * random.random()
        return delay + jitter
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limit hit, retrying in {retry_state.next_action.sleep} seconds..."
        )
    )
    def _call_llm_with_retry(self, chain, inputs):
        """Call LLM with retry logic for rate limits"""
        try:
            return chain.invoke(inputs)
        except Exception as e:
            # Check if it's a rate limit error from the API
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "too many requests" in error_msg or "429" in error_msg:
                logger.warning(f"Rate limit error: {e}")
                raise RateLimitError(f"Rate limit exceeded: {e}")
            else:
                # For other errors, just raise them normally
                raise
    
    def extract_from_text(self, 
                         text: str,
                         use_structured_output: bool = True) -> ExtractedData:
        """Extract information from text with rate limit handling"""
        start_time = time.time()
        
        try:
            if use_structured_output:
                # Use structured output with Pydantic
                output_model = self.create_output_model()
                parser = PydanticOutputParser(pydantic_object=output_model)
                
                prompt = self.create_extraction_prompt()
                prompt = prompt.partial(format_instructions=parser.get_format_instructions())

                chain = prompt | self.llm | parser
                result = self._call_llm_with_retry(chain, {"text": text})
                extracted_fields = result.dict()
            else:
                # Use unstructured output
                prompt = self.create_extraction_prompt()
                chain = prompt | self.llm | StrOutputParser()
                result = self._call_llm_with_retry(chain, {"text": text})
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
                    "model_name": self.llm.model,
                    "retries": getattr(result, "_retry_state", {}).get("attempt_number", 1) - 1
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
    
    # def _parse_unstructured_output(self, output: str) -> Dict:
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
            
    def extract_from_file(self, 
                         file_path: Union[str, Path],
                         use_structured_output: bool = True) -> ExtractedData:
        """Extract information from a file"""
        path = Path(file_path) if isinstance(file_path, str) else file_path
        
        if not path.exists():
            return ExtractedData(
                document_type=self.document_type,
                raw_text="",
                extracted_fields={},
                validation_errors=[f"File not found: {path}"],
                processing_time=0
            )
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            return self.extract_from_text(text, use_structured_output)
        
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return ExtractedData(
                document_type=self.document_type,
                raw_text="",
                extracted_fields={},
                validation_errors=[f"File error: {str(e)}"],
                processing_time=0
            )