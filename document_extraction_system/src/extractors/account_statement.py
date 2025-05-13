# src/extractors/account_statement.py
from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from ..base.extractor import BaseDocumentExtractor
from ..base.models import DocumentType


class AccountStatementExtractor(BaseDocumentExtractor):
    """Extractor for account statements"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(DocumentType.ACCOUNT_STATEMENT, api_key)
    
    def create_output_model(self) -> Type[BaseModel]:
        """Create Pydantic model for account statement"""
        class Transaction(BaseModel):
            date: str = Field(description="Date de l'opération")
            description: str = Field(description="Description de l'opération")
            debit: Optional[float] = Field(None, description="Montant au débit")
            credit: Optional[float] = Field(None, description="Montant au crédit")
            operations_count: Optional[int] = Field(None, description="Nombre d'opérations")
            
        class AccountStatement(BaseModel):
            account_holder_name: str = Field(description="Nom du titulaire du compte")
            account_holder_address: Optional[str] = Field(None, description="Adresse du titulaire")
            account_number: str = Field(description="Numéro de compte (RIB/BBAN)")
            currency: str = Field(description="Devise du compte")
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
        You are an expert at extracting information from bank account statements (relevé de compte).
        Extract the following information from the document text:

        - Account holder name (company name or person's first and last name)
        - Account holder address
        - Account number (RIB/BBAN)
        - Currency (likely "Dirham marocain")
        - Statement period (start and end dates)
        - Opening balance (Solde de départ)
        - Closing balance (Nouveau Solde)
        - Total debits (Au débit)
        - Total credits (Au crédit)
        - List of transactions with:
          - Date (Date Opération)
          - Description
          - Debit amount (if applicable, shown as negative)
          - Credit amount (if applicable)
          - Number of operations (Nombre d'opérations)

        The document may contain both French and Arabic text.
        
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
            'account_holder_name', 'account_number', 'period_start', 'period_end',
            'opening_balance', 'closing_balance', 'total_debits', 'total_credits'
        ]
        
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate balance consistency
        if 'transactions' in data and data['transactions']:
            calculated_balance = data.get('opening_balance', 0)
            total_debits = 0
            total_credits = 0
            
            for txn in data['transactions']:
                if txn.get('credit'):
                    total_credits += txn['credit']
                    calculated_balance += txn['credit']
                if txn.get('debit'):
                    total_debits += abs(txn['debit'])  # Debits are often negative
                    calculated_balance -= abs(txn['debit'])
            
            # Check if calculated balance matches closing balance (with tolerance)
            if abs(calculated_balance - data.get('closing_balance', 0)) > 0.01:
                errors.append("Calculated balance doesn't match closing balance")
            
            # Check if totals match
            if abs(total_debits - data.get('total_debits', 0)) > 0.01:
                errors.append("Sum of transaction debits doesn't match total debits")
            
            if abs(total_credits - data.get('total_credits', 0)) > 0.01:
                errors.append("Sum of transaction credits doesn't match total credits")
        
        # Validate date formats
        import re
        date_pattern = r'\d{2}/\d{2}/\d{4}'
        
        if data.get('period_start') and not re.match(date_pattern, data['period_start']):
            errors.append("Invalid date format for period_start")
        
        if data.get('period_end') and not re.match(date_pattern, data['period_end']):
            errors.append("Invalid date format for period_end")
        
        return errors