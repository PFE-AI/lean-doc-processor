from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from .base_extraction_system import (
    BaseDocumentExtractor,
    DocumentType,
    DocumentExtractorFactory
)


class AccountStatementExtractor(BaseDocumentExtractor):
    """Extractor for account statements"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(DocumentType.ACCOUNT_STATEMENT, api_key)
    
    def create_output_model(self) -> Type[BaseModel]:
        """Create Pydantic model for account statement"""
        class Transaction(BaseModel):
            date: str = Field(description="Date de transaction")
            description: str = Field(description="Description")
            debit: Optional[float] = Field(None, description="Montant débit")
            credit: Optional[float] = Field(None, description="Montant crédit")
            balance: float = Field(description="Solde")
            
        class AccountStatement(BaseModel):
            account_number: str = Field(description="Numéro de compte")
            account_holder: str = Field(description="Titulaire du compte")
            period_start: str = Field(description="Date de début de période")
            period_end: str = Field(description="Date de fin de période")
            opening_balance: float = Field(description="Solde d'ouverture")
            closing_balance: float = Field(description="Solde de clôture")
            total_debits: float = Field(description="Total des débits")
            total_credits: float = Field(description="Total des crédits")
            transactions: List[Transaction] = Field(description="Liste des transactions")
            
        return AccountStatement
    
    def create_extraction_prompt(self) -> PromptTemplate:
        """Create prompt for account statement extraction"""
        template = """
        You are an expert at extracting information from bank account statements.
        Extract the following information from the document text:

        - Account number
        - Account holder name
        - Statement period (start and end dates)
        - Opening balance
        - Closing balance
        - Total debits
        - Total credits
        - List of transactions with:
          - Date
          - Description
          - Debit amount (if applicable)
          - Credit amount (if applicable)
          - Balance after transaction

        Document text:
        {text}

        {format_instructions}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["text"],
            partial_variables={}
        )
    
    def validate_extracted_data(self, data: Dict) -> List[str]:
        """Validate account statement data"""
        errors = []
        
        # Check required fields
        required_fields = [
            'account_number', 'account_holder', 'period_start', 'period_end',
            'opening_balance', 'closing_balance', 'total_debits', 'total_credits'
        ]
        
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate balance consistency
        if 'transactions' in data and data['transactions']:
            calculated_balance = data.get('opening_balance', 0)
            for txn in data['transactions']:
                if txn.get('credit'):
                    calculated_balance += txn['credit']
                if txn.get('debit'):
                    calculated_balance -= txn['debit']
            
            # Allow for small rounding differences
            if abs(calculated_balance - data.get('closing_balance', 0)) > 0.01:
                errors.append("Calculated balance doesn't match closing balance")
        
        # Validate that total debits and credits match transaction sums
        if 'transactions' in data and data['transactions']:
            total_debits = sum(txn.get('debit', 0) for txn in data['transactions'])
            total_credits = sum(txn.get('credit', 0) for txn in data['transactions'])
            
            if abs(total_debits - data.get('total_debits', 0)) > 0.01:
                errors.append("Sum of transaction debits doesn't match total debits")
            
            if abs(total_credits - data.get('total_credits', 0)) > 0.01:
                errors.append("Sum of transaction credits doesn't match total credits")
        
        return errors


# Register the new extractor
DocumentExtractorFactory.register_extractor(
    DocumentType.ACCOUNT_STATEMENT,
    AccountStatementExtractor
)


# Generic extractor for unknown document types
class GenericDocumentExtractor(BaseDocumentExtractor):
    """Generic extractor for unknown document types"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(DocumentType.GENERIC, api_key)
    
    def create_output_model(self) -> Type[BaseModel]:
        """Create generic Pydantic model"""
        class GenericDocument(BaseModel):
            title: Optional[str] = Field(None, description="Document title")
            date: Optional[str] = Field(None, description="Document date")
            reference_number: Optional[str] = Field(None, description="Reference number")
            key_values: Dict[str, str] = Field(description="Key-value pairs found in document")
            tables: Optional[List[List[str]]] = Field(None, description="Tables found in document")
            summary: str = Field(description="Document summary")
            
        return GenericDocument
    
    def create_extraction_prompt(self) -> PromptTemplate:
        """Create prompt for generic document extraction"""
        template = """
        You are an expert at extracting information from documents.
        This is a generic document. Extract any important information you can find:

        - Document title or type
        - Date
        - Reference number or ID
        - Key-value pairs (like field names and their values)
        - Any tables present
        - A brief summary of the document

        Document text:
        {text}

        {format_instructions}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["text"],
            partial_variables={}
        )
    
    def validate_extracted_data(self, data: Dict) -> List[str]:
        """Minimal validation for generic documents"""
        errors = []
        
        if not data.get('summary'):
            errors.append("Document summary is required")
        
        return errors


# Register the generic extractor
DocumentExtractorFactory.register_extractor(
    DocumentType.GENERIC,
    GenericDocumentExtractor
)


# Example of how to use the extended system
def example_new_document_types():
    """Example usage with new document types"""
    import os
    import json
    from .base_extraction_system import DocumentExtractionPipeline
    
    # Initialize pipeline
    api_key = os.getenv("MISTRAL_API_KEY")
    pipeline = DocumentExtractionPipeline(api_key=api_key)
    
    # Extract from account statement
    account_statement_path = "path/to/account_statement.pdf"
    result = pipeline.extract_from_document(
        account_statement_path,
        DocumentType.ACCOUNT_STATEMENT,
        use_structured_output=True
    )
    
    print("Account Statement Extraction:")
    print(f"Account Number: {result.extracted_fields.get('account_number')}")
    print(f"Account Holder: {result.extracted_fields.get('account_holder')}")
    print(f"Period: {result.extracted_fields.get('period_start')} to {result.extracted_fields.get('period_end')}")
    print(f"Opening Balance: {result.extracted_fields.get('opening_balance')}")
    print(f"Closing Balance: {result.extracted_fields.get('closing_balance')}")
    print(f"Number of Transactions: {len(result.extracted_fields.get('transactions', []))}")
    
    # Auto-detect document type
    unknown_document_path = "path/to/unknown_document.pdf"
    detected_type = pipeline.auto_detect_document_type(unknown_document_path)
    print(f"\nDetected document type: {detected_type}")
    
    # Extract from unknown document
    result = pipeline.extract_from_document(
        unknown_document_path,
        detected_type,
        use_structured_output=True
    )
    
    print(f"Extracted Fields: {json.dumps(result.extracted_fields, indent=2)}")


if __name__ == "__main__":
    example_new_document_types()