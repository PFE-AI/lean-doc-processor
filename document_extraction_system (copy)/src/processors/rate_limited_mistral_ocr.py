# src/processors/rate_limited_mistral_ocr.py
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
import logging
import os
import base64
import mimetypes
import time
import random
import requests
from mistralai import Mistral

logger = logging.getLogger(__name__)

class RateLimitedMistralOCRProcessor:
    """
    Document processor that uses Mistral AI's OCR capabilities with rate limiting support
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "mistral-ocr-latest", 
                 max_retries: int = 5,
                 initial_backoff: float = 1.0,
                 backoff_factor: float = 2.0,
                 jitter: float = 0.1):
        """
        Initialize the Mistral OCR document processor with rate limiting
        
        Args:
            api_key: Mistral API key. If None, will use MISTRAL_API_KEY env var
            model: Mistral model to use (must be a dedicated OCR model)
            max_retries: Maximum number of retries for rate-limited requests
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Multiplier for exponential backoff
            jitter: Random jitter factor to add to backoff (as a fraction)
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY env var or pass to constructor.")
        
        self.model = model
        self.client = Mistral(api_key=self.api_key)
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        
        logger.info(f"Initialized RateLimitedMistralOCRProcessor with model: {model}")
    
    def _execute_with_backoff(self, operation_func: Callable, *args, **kwargs):
        """
        Execute an operation with exponential backoff for rate limiting
        
        Args:
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation function
        
        Raises:
            Exception: If max retries exceeded or non-rate-limit error occurs
        """
        retries = 0
        backoff_time = self.initial_backoff
        
        while True:
            try:
                return operation_func(*args, **kwargs)
            
            except requests.exceptions.HTTPError as e:
                # Check if it's a rate limit error (429)
                if e.response.status_code == 429:
                    if retries >= self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) exceeded for rate limiting")
                        raise
                    
                    # Calculate backoff with jitter
                    jitter_amount = backoff_time * self.jitter * random.uniform(-1, 1)
                    sleep_time = backoff_time + jitter_amount
                    
                    logger.warning(f"Rate limit hit. Retrying in {sleep_time:.2f}s (retry {retries + 1}/{self.max_retries})")
                    time.sleep(sleep_time)
                    
                    # Increase backoff for next attempt
                    backoff_time *= self.backoff_factor
                    retries += 1
                else:
                    # Non-rate-limit API error
                    logger.error(f"HTTP error: {e}")
                    raise
            
            except Exception as e:
                # Check for rate limit error messages in other exception types
                error_message = str(e).lower()
                if "429" in error_message or "rate limit" in error_message:
                    if retries >= self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) exceeded for rate limiting")
                        raise
                    
                    # Calculate backoff with jitter
                    jitter_amount = backoff_time * self.jitter * random.uniform(-1, 1)
                    sleep_time = backoff_time + jitter_amount
                    
                    logger.warning(f"Rate limit detected. Retrying in {sleep_time:.2f}s (retry {retries + 1}/{self.max_retries})")
                    time.sleep(sleep_time)
                    
                    # Increase backoff for next attempt
                    backoff_time *= self.backoff_factor
                    retries += 1
                else:
                    # Handle other exceptions
                    logger.error(f"Error during API call: {e}")
                    raise
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a document file with OCR, handling rate limits
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            file_path = Path(file_path)
            
            # Validate file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Get file MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/octet-stream"
            
            # Determine if we're dealing with PDF or image
            document_type = "document_url" if mime_type == "application/pdf" else "image_url"
            
            # Encode file to base64
            with open(file_path, "rb") as f:
                file_content = f.read()
                base64_content = base64.b64encode(file_content).decode('utf-8')
            
            logger.info(f"Processing document: {file_path}")
            
            # Prepare data URL based on mime type
            data_url = f"data:{mime_type};base64,{base64_content}"
            
            # Call the OCR API with rate limit handling
            def ocr_operation():
                return self.client.ocr.process(
                    model=self.model,
                    document={
                        "type": document_type,
                        document_type: data_url
                    },
                    include_image_base64=True
                )
            
            ocr_response = self._execute_with_backoff(ocr_operation)

            # Create result dictionary from the OCR response
            extracted_text = ""
            if hasattr(ocr_response, "content"):
                # Handle direct content attribute
                extracted_text = ocr_response.content
            elif hasattr(ocr_response, "pages") and ocr_response.pages:
                # Extract text from all pages if available
                page_texts = []
                for page in ocr_response.pages:
                    # Check for different possible field names (content or markdown)
                    if hasattr(page, "content"):
                        page_texts.append(page.content)
                    elif hasattr(page, "markdown"):
                        page_texts.append(page.markdown)
                    # If page is a dictionary (not an object)
                    elif isinstance(page, dict):
                        if "content" in page:
                            page_texts.append(page["content"])
                        elif "markdown" in page:
                            page_texts.append(page["markdown"])
                
                extracted_text = "\n\n".join(page_texts)

            result = {
                "text": extracted_text,
                "quality_score": self._calculate_quality_score(extracted_text),
                "metadata": {
                    "model_used": self.model,
                    "file_size": file_path.stat().st_size,
                    "file_type": mime_type,
                    "additional_info": self._extract_metadata(ocr_response)
                }
            }
            print(ocr_response)
            print(result)
            logger.info(f"Successfully processed document with {len(result['text'])} characters")
            return result
                
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            # Return a minimal result with error information
            return {
                "text": f"Error processing document: {str(e)}",
                "quality_score": 0.0,
                "error": str(e),
                "metadata": {
                    "file_path": str(file_path),
                    "status": "failed"
                }
            }
    
    def process_multiple_files(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Process multiple document files with OCR, handling rate limits
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                # Add empty result with error info
                results.append({
                    "text": "",
                    "quality_score": 0.0,
                    "error": str(e),
                    "metadata": {
                        "file_path": str(file_path),
                        "status": "failed"
                    }
                })
        
        return results
    
    def _calculate_quality_score(self, text: str) -> float:
        """
        Calculate a simple OCR quality score based on text content
        
        Args:
            text: Extracted text from OCR
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not text:
            return 0.0
        
        # Simple heuristics for quality scoring:
        # 1. Length of text (longer is generally better for documents)
        # 2. Ratio of alphanumeric characters to total length
        # 3. Presence of word boundaries
        
        text_length = len(text)
        alnum_count = sum(c.isalnum() for c in text)
        word_count = len(text.split())
        
        # Calculate alphanumeric ratio (penalize if too many non-text chars)
        alnum_ratio = alnum_count / text_length if text_length > 0 else 0.0
        
        # Calculate word density (penalize if too few words for length)
        word_density = min(1.0, word_count / (text_length / 50)) if text_length > 0 else 0.0
        
        # Weighted score
        score = (0.4 * min(1.0, text_length / 1000)) + (0.3 * alnum_ratio) + (0.3 * word_density)
        
        return min(1.0, max(0.0, score))
    
    def _extract_metadata(self, response: Any) -> Dict[str, Any]:
        """
        Extract any additional metadata from the response
        
        Args:
            response: OCR response
            
        Returns:
            Dictionary with additional metadata
        """
        # Extract various attributes from OCR response
        metadata = {}
        
        # Try to extract common attributes from the response
        for attr in ['id', 'model', 'object', 'usage', 'created_at']:
            if hasattr(response, attr):
                metadata[attr] = getattr(response, attr)
                
        # Get model information if available in different formats
        if hasattr(response, 'model_name'):
            metadata['model_name'] = response.model_name
        elif hasattr(response, 'model'):
            metadata['model_name'] = response.model
                
        # If there are any page-specific information
        if hasattr(response, 'pages'):
            metadata['page_count'] = len(response.pages)
            
        return metadata
        
    # Alternative method to upload and process a file
    def upload_and_process(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Upload a file to Mistral and process it with OCR, handling rate limits
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            file_path = Path(file_path)
            
            # Validate file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Upload the file with rate limit handling
            def upload_operation():
                return self.client.files.upload(
                    file={
                        "file_name": file_path.name,
                        "content": open(file_path, "rb"),
                    },
                    purpose="ocr"
                )
            
            uploaded_file = self._execute_with_backoff(upload_operation)
            
            # Get a signed URL for the uploaded file
            def signed_url_operation():
                return self.client.files.get_signed_url(file_id=uploaded_file.id)
            
            signed_url = self._execute_with_backoff(signed_url_operation)
            
            # Process the file with OCR
            def ocr_operation():
                return self.client.ocr.process(
                    model=self.model,
                    document={
                        "type": "document_url",
                        "document_url": signed_url.url
                    }
                )
            
            ocr_response = self._execute_with_backoff(ocr_operation)
            
            # Extract text from the OCR response
            extracted_text = ""
            if hasattr(ocr_response, "content"):
                extracted_text = ocr_response.content
            elif hasattr(ocr_response, "pages") and ocr_response.pages:
                # Extract text from all pages if available
                extracted_text = "\n\n".join([page.get("content", "") for page in ocr_response.pages if "content" in page])
            
            # Create result dictionary
            result = {
                "text": extracted_text,
                "quality_score": self._calculate_quality_score(extracted_text),
                "metadata": {
                    "model_used": self.model,
                    "file_id": uploaded_file.id,
                    "file_name": file_path.name,
                    "additional_info": self._extract_metadata(ocr_response)
                }
            }
            
            logger.info(f"Successfully processed document with {len(result['text'])} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            # Return a minimal result with error information
            return {
                "text": f"Error processing document: {str(e)}",
                "quality_score": 0.0,
                "error": str(e),
                "metadata": {
                    "file_path": str(file_path),
                    "status": "failed"
                }
            }