# src/processors/mistral_ocr.py
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
import os
import base64
import mimetypes
from mistralai import Mistral

logger = logging.getLogger(__name__)

class MistralOCRDocumentProcessor:
    """
    Document processor that uses Mistral AI's OCR capabilities
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-ocr-latest"):
        """
        Initialize the Mistral OCR document processor
        
        Args:
            api_key: Mistral API key. If None, will use MISTRAL_API_KEY env var
            model: Mistral model to use (must be a dedicated OCR model)
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY env var or pass to constructor.")
        
        self.model = model
        self.client = Mistral(api_key=self.api_key)
        logger.info(f"Initialized MistralOCRDocumentProcessor with model: {model}")
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a document file with OCR
        
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
            
            # Call the OCR API directly
            ocr_response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": document_type,
                    document_type: data_url
                },
                include_image_base64=True
            )
            
            # Create result dictionary from the OCR response
            # Handle the actual OCR response format - the text might be in 'content' or directly in the response
            extracted_text = ""
            if hasattr(ocr_response, "content"):
                extracted_text = ocr_response.content
            elif hasattr(ocr_response, "pages") and ocr_response.pages:
                # Extract text from all pages if available
                extracted_text = "\n\n".join([page.get("content", "") for page in ocr_response.pages if "content" in page])
            
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
        Process multiple document files with OCR
        
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
        
        # If there are any page-specific information
        if hasattr(response, 'pages'):
            metadata['page_count'] = len(response.pages)
            
        return metadata
        
    # Alternative method to upload and process a file
    def upload_and_process(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Upload a file to Mistral and process it with OCR
        
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
            
            # Upload the file
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": file_path.name,
                    "content": open(file_path, "rb"),
                },
                purpose="ocr"
            )
            
            # Get a signed URL for the uploaded file
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
            
            # Process the file with OCR
            ocr_response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": signed_url.url
                }
            )
            
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