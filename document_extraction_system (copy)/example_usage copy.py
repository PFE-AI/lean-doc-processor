from dotenv import load_dotenv
import os
import json
from pathlib import Path
from src.pipeline import DocumentExtractionPipeline
from src.base.models import DocumentType
from src.extractors.account_statement import AccountStatementExtractor

load_dotenv()

def main():
    # Initialize the pipeline
    api_key = os.environ.get("MISTRAL_API_KEY")
    pipeline = DocumentExtractionPipeline(api_key=api_key)
    
    # Example 1: Extract from a single document with auto-detection
    print("=== Example 1: Single document with auto-detection ===")
    document_path = "path/to/account_statement.pdf"
    
    if Path(document_path).exists():
        result = pipeline.extract_from_document(document_path)
        print(f"Document Type: {result.document_type.value}")
        print(f"Extracted Fields: {json.dumps(result.extracted_fields, indent=2, ensure_ascii=False)}")
        print(f"Validation Errors: {result.validation_errors}")
        print(f"OCR Quality Score: {result.ocr_quality_score}")
        print()
    
    # Example 2: Extract from a specific document type
    print("=== Example 2: Specific document type ===")
    receipt_path = "path/to/bill_payment_receipt.pdf"
    
    if Path(receipt_path).exists():
        result = pipeline.extract_from_document(
            receipt_path,
            document_type=DocumentType.BILL_PAYMENT_RECEIPT_A4
        )
        
        # Print extracted bill details
        if 'bills' in result.extracted_fields:
            print("Paid Bills:")
            for bill in result.extracted_fields['bills']:
                print(f"  - {bill['number']}: {bill['description']} - {bill['amount']}")
        
        print(f"Total Amount: {result.extracted_fields.get('total_amount')}")
        print()
    
    # Example 3: Batch processing
    print("=== Example 3: Batch processing ===")
    documents = [
        {
            "path": "path/to/account_statement.pdf",
            "type": DocumentType.ACCOUNT_STATEMENT
        },
        {
            "path": "path/to/commission_statement.pdf",
            "type": DocumentType.COMMISSION_STATEMENT
        },
        {
            "path": "path/to/receipt_80mm.pdf",
            "type": DocumentType.BILL_PAYMENT_RECEIPT_80MM
        }
    ]
    
    results = pipeline.batch_extract(documents)
    
    for i, result in enumerate(results):
        print(f"Document {i+1}: {result.document_type.value}")
        print(f"  Validation: {'PASSED' if not result.validation_errors else 'FAILED'}")
        print(f"  Processing Time: {result.processing_time:.2f}s")
        print()
    
    # Example 4: Validation report
    print("=== Example 4: Validation report ===")
    if results:
        validation_report = pipeline.validate_extraction(results[0])
        print(f"Validation Report:")
        print(f"  Is Valid: {validation_report['is_valid']}")
        print(f"  Errors: {validation_report['errors']}")
        print(f"  Warnings: {validation_report['warnings']}")
        print(f"  Field Completeness:")
        for field, status in validation_report['field_completeness'].items():
            print(f"    - {field}: {status}")
    
    # Example 5: Export results
    print("=== Example 5: Export results ===")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    if results:
        for i, result in enumerate(results):
            output_file = output_dir / f"extracted_{i+1}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'document_type': result.document_type.value,
                    'extracted_fields': result.extracted_fields,
                    'validation_errors': result.validation_errors,
                    'metadata': result.metadata,
                    'processing_time': result.processing_time,
                    'ocr_quality_score': result.ocr_quality_score
                }, f, indent=2, ensure_ascii=False)
            print(f"Exported to: {output_file}")


def test_specific_extractor():
    
    api_key = os.environ.get("MISTRAL_API_KEY")
    extractor = AccountStatementExtractor(api_key=api_key)
    
    # Sample text (you would get this from OCR)
    sample_text = """
    RELEVÉ DE COMPTE
    
    Nom: ENTREPRISE ABC
    Compte: 123456789012
    Période: du 01/01/2024 au 31/01/2024
    
    Solde de départ au 01/01/2024: 10,000.00
    
    OPERATIONS:
    05/01/2024 - Virement client - Crédit: 5,000.00
    10/01/2024 - Paiement fournisseur - Débit: 2,000.00
    
    Total débits: 2,000.00
    Total crédits: 5,000.00
    
    Nouveau solde au 31/01/2024: 13,000.00
    """
    
    result = extractor.extract_from_text(sample_text)
    print(json.dumps(result.extracted_fields, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # main()
    test_specific_extractor()