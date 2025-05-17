from dotenv import load_dotenv
import os
import json
from pathlib import Path
from src.pipeline import DocumentExtractionPipeline
from src.base.models import DocumentType

load_dotenv()

def main():
    # Initialize the pipeline
    api_key = os.environ.get("MISTRAL_API_KEY")
    pipeline = DocumentExtractionPipeline(api_key=api_key)
    
    # Example 1: Extract from a single document with auto-detection
    print("=== Example 1: Single document with auto-detection ===")
    document_path = "/home/anas/Desktop/lean-doc-processor/data/training_data/validation/commission_statements/commission_statement_invoice_90_page_1_rotated_processed.png"
    
    if Path(document_path).exists():
        result = pipeline.extract_from_document(document_path, document_type = DocumentType.COMMISSION_STATEMENT)
        print(f"Document Type: {result.document_type.value}")
        print(f"Extracted Fields: {json.dumps(result.extracted_fields, indent=2, ensure_ascii=False)}")
        print(f"Validation Errors: {result.validation_errors}")
        print(f"OCR Quality Score: {result.ocr_quality_score}")
        print()
   

if __name__ == "__main__":
    # main()
    main()