# src/pipeline.py
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
import os

from .base.models import DocumentType, ExtractedData
from .base.factory import DocumentExtractorFactory
from .processors.mistral_ocr import MistralOCRDocumentProcessor
from .extractors.account_statement import AccountStatementExtractor
from .extractors.commission_statement import CommissionStatementExtractor
from .extractors.bill_payment_receipt import (
    BillPaymentReceiptA4Extractor,
    BillPaymentReceipt80mmExtractor
)

logger = logging.getLogger(__name__)


class DocumentExtractionPipeline:
    """Main pipeline for document extraction"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.ocr_processor = MistralOCRDocumentProcessor(api_key=self.api_key)
        self.factory = DocumentExtractorFactory()
        
        # Register all extractors
        self._register_extractors()
    
    def _register_extractors(self):
        """Register all available extractors"""
        self.factory.register_extractor(
            DocumentType.ACCOUNT_STATEMENT, 
            AccountStatementExtractor
        )
        self.factory.register_extractor(
            DocumentType.COMMISSION_STATEMENT, 
            CommissionStatementExtractor
        )
        self.factory.register_extractor(
            DocumentType.BILL_PAYMENT_RECEIPT_A4, 
            BillPaymentReceiptA4Extractor
        )
        self.factory.register_extractor(
            DocumentType.BILL_PAYMENT_RECEIPT_80MM, 
            BillPaymentReceipt80mmExtractor
        )
    
    def extract_from_document(self,
                             document_path: Union[str, Path],
                             document_type: Optional[DocumentType] = None,
                             use_structured_output: bool = True) -> ExtractedData:
        """
        Extract data from a document
        
        Args:
            document_path: Path to the document
            document_type: Type of document. If None, will attempt to auto-detect
            use_structured_output: Whether to use structured output
            
        Returns:
            ExtractedData object containing extracted information
        """
        document_path = Path(document_path)
        
        # Process document with OCR
        logger.info(f"Processing document with OCR: {document_path}")
        ocr_result = self.ocr_processor.process_file(document_path)
        
        # Auto-detect document type if not provided
        if document_type is None:
            document_type = self.auto_detect_document_type(ocr_result['text'])
            logger.info(f"Auto-detected document type: {document_type}")
        
        # Create appropriate extractor
        extractor = self.factory.create_extractor(document_type, self.api_key)
        
        # Extract data
        logger.info(f"Extracting data using {document_type.value} extractor")
        extracted_data = extractor.extract_from_text(
            ocr_result['text'], 
            use_structured_output
        )
        
        # Add OCR metadata to extracted data
        extracted_data.metadata.update({
            'source_file': str(document_path),
            'ocr_quality_score': ocr_result.get('quality_score'),
            'ocr_metadata': ocr_result.get('metadata', {})
        })
        extracted_data.ocr_quality_score = ocr_result.get('quality_score')
        
        return extracted_data
    
    def batch_extract(self,
                     documents: List[Dict[str, Union[str, DocumentType]]],
                     use_structured_output: bool = True) -> List[ExtractedData]:
        """
        Extract data from multiple documents
        
        Args:
            documents: List of dictionaries with 'path' and optionally 'type'
            use_structured_output: Whether to use structured output
            
        Returns:
            List of ExtractedData objects
        """
        results = []
        
        for doc in documents:
            try:
                result = self.extract_from_document(
                    doc['path'],
                    doc.get('type'),
                    use_structured_output
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing document {doc['path']}: {e}")
                # Create a failed result
                failed_result = ExtractedData(
                    document_type=doc.get('type', DocumentType.GENERIC),
                    raw_text='',
                    extracted_fields={},
                    validation_errors=[f"Processing error: {str(e)}"],
                    metadata={'source_file': doc['path']}
                )
                results.append(failed_result)
        
        return results
    
    def auto_detect_document_type(self, text: str) -> DocumentType:
        """
        Auto-detect document type from content
        
        Args:
            text: OCR extracted text
            
        Returns:
            Detected DocumentType
        """
        text_lower = text.lower()
        
        # Check for account statement indicators
        if 'relevé de compte' in text_lower or 'كشف الحساب' in text:
            if 'solde de départ' in text_lower or 'nouveau solde' in text_lower:
                return DocumentType.ACCOUNT_STATEMENT
        
        # Check for commission statement indicators
        if 'commission' in text_lower and ('total' in text_lower or 'montant' in text_lower):
            return DocumentType.COMMISSION_STATEMENT
        
        # Check for bill payment receipt indicators
        if 'reçu de paiement' in text_lower:
            # Differentiate between A4 and 80mm formats
            # This is a heuristic - you might need to adjust based on actual differences
            # For now, we'll use a simple check based on content density
            lines = text.split('\n')
            avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
            
            # 80mm receipts tend to have shorter lines
            if avg_line_length < 40:
                return DocumentType.BILL_PAYMENT_RECEIPT_80MM
            else:
                return DocumentType.BILL_PAYMENT_RECEIPT_A4
        
        # Default to generic if no specific type detected
        return DocumentType.GENERIC
    
    def validate_extraction(self, extracted_data: ExtractedData) -> Dict[str, any]:
        """
        Perform additional validation on extracted data
        
        Args:
            extracted_data: ExtractedData object to validate
            
        Returns:
            Validation report dictionary
        """
        report = {
            'document_type': extracted_data.document_type.value,
            'is_valid': len(extracted_data.validation_errors) == 0,
            'errors': extracted_data.validation_errors,
            'warnings': [],
            'field_completeness': {},
            'ocr_quality': extracted_data.ocr_quality_score
        }
        
        # Check field completeness
        fields = extracted_data.extracted_fields
        expected_fields = self._get_expected_fields(extracted_data.document_type)
        
        for field in expected_fields:
            if field in fields and fields[field] is not None:
                report['field_completeness'][field] = 'present'
            else:
                report['field_completeness'][field] = 'missing'
                report['warnings'].append(f"Expected field '{field}' is missing")
        
        # Check OCR quality
        if extracted_data.ocr_quality_score and extracted_data.ocr_quality_score < 0.7:
            report['warnings'].append(f"Low OCR quality score: {extracted_data.ocr_quality_score:.2f}")
        
        return report
    
    def _get_expected_fields(self, document_type: DocumentType) -> List[str]:
        """Get expected fields for a document type"""
        expected_fields = {
            DocumentType.ACCOUNT_STATEMENT: [
                'account_holder_name', 'account_number', 'period_start', 
                'period_end', 'opening_balance', 'closing_balance', 
                'total_debits', 'total_credits', 'transactions'
            ],
            DocumentType.COMMISSION_STATEMENT: [
                'period', 'agent_code', 'agent_name', 'total_transactions',
                'total_amount', 'total_commission', 'net_commission'
            ],
            DocumentType.BILL_PAYMENT_RECEIPT_A4: [
                'merchant_name', 'payment_date', 'receipt_number', 
                'creditor_tx_number', 'inwi_tx_number', 'agent_code',
                'bills', 'management_fees', 'total_amount'
            ],
            DocumentType.BILL_PAYMENT_RECEIPT_80MM: [
                'merchant_name', 'payment_date', 'receipt_number', 
                'creditor_tx_number', 'inwi_tx_number', 'agent_code',
                'bills', 'management_fees', 'total_amount'
            ]
        }
        
        return expected_fields.get(document_type, [])