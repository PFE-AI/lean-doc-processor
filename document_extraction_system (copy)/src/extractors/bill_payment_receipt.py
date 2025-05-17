# src/extractors/bill_payment_receipt.py
from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from ..base.extractor import BaseDocumentExtractor
from ..base.models import DocumentType


class BillPaymentReceiptExtractor(BaseDocumentExtractor):
    """Extractor for bill payment receipts (both A4 and 80mm)"""
    
    def __init__(self, api_key: Optional[str] = None, is_80mm: bool = False):
        doc_type = DocumentType.BILL_PAYMENT_RECEIPT_80MM if is_80mm else DocumentType.BILL_PAYMENT_RECEIPT_A4
        super().__init__(doc_type, api_key)
        self.is_80mm = is_80mm
    
    def create_output_model(self) -> Type[BaseModel]:
        """Create Pydantic model for bill payment receipt"""
        class BillDetail(BaseModel):
            number: str = Field(description="Référence de la facture")
            description: str = Field(description="Description")
            amount: str = Field(description="Montant (DH TTC)")
            
        class AdditionalInfo(BaseModel):
            key: str = Field(description="Nom du champ")
            value: str = Field(description="Valeur du champ")
            
        class BillPaymentReceipt(BaseModel):
            merchant_name: str = Field(description="Nom du marchand")
            payment_date: str = Field(description="Date de paiement")
            receipt_number: str = Field(description="N° reçu")
            creditor_tx_number: str = Field(description="N° Tx créancier")
            inwi_tx_number: str = Field(description="N° Tx inwi money")
            agent_code: str = Field(description="Code agent")
            reference: Optional[str] = Field(None, description="Référence/Immatriculation")
            bills: List[BillDetail] = Field(description="Liste des factures payées")
            management_fees: str = Field(description="Frais de gestion")
            stamp_duty: Optional[float] = Field(None, description="Droits de timbre")
            total_amount: str = Field(description="Total (DH TTC)")
            additional_info: Optional[List[AdditionalInfo]] = Field(None, description="Informations supplémentaires")
            duplicate_date: Optional[str] = Field(None, description="Date du duplicata")
            marketing_message: Optional[str] = Field(None, description="Message marketing")
            
        return BillPaymentReceipt
    
    def create_extraction_prompt(self) -> PromptTemplate:
        """Create prompt for bill payment receipt extraction"""
        template = f"""
        You are an expert at extracting information from bill payment receipts {"(80mm format)" if self.is_80mm else "(A4 format)"}.
        This is a REÇU DE PAIEMENT document.
        
        Extract the following information from the document text:

        - Merchant name (shown at the top)
        - Payment date (Date de paiement)
        - Receipt number (N° reçu)
        - Creditor transaction number (N° Tx créancier)
        - Inwi transaction number (N° Tx inwi money)
        - Agent code (Code agent)
        - Reference/Registration (Référence/Immatriculation) if present
        - List of paid bills containing:
          - Reference number
          - Description
          - Amount (in DH TTC)
        - Management fees (Frais de gestion)
        - Stamp duty (Droits de timbre) if present
        - Total amount (Total DH TTC)
        - Any additional information fields
        - If it's a duplicate (DUPLICATA), extract the duplicate date
        - Marketing message if present

        Note: Amounts should be extracted as strings to preserve formatting.
        
        Document text:
        {{text}}

        {{format_instructions}}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["text"],
            partial_variables={}
        )
    
    def validate_extracted_data(self, data: Dict) -> List[str]:
        """Validate bill payment receipt data"""
        errors = []
        
        # Check required fields
        required_fields = [
            'merchant_name', 'payment_date', 'receipt_number', 
            'creditor_tx_number', 'inwi_tx_number', 'agent_code',
            'management_fees', 'total_amount'
        ]
        
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate that bills list exists and is not empty
        if 'bills' not in data or not data['bills']:
            errors.append("At least one bill must be present")
        else:
            # Validate each bill has required fields
            for i, bill in enumerate(data['bills']):
                if not bill.get('number'):
                    errors.append(f"Bill {i+1} missing reference number")
                if not bill.get('description'):
                    errors.append(f"Bill {i+1} missing description")
                if not bill.get('amount'):
                    errors.append(f"Bill {i+1} missing amount")
        
        # Validate amount format (should be strings, but can check if they're numeric)
        import re
        amount_pattern = r'^-?\d+(?:[.,]\d{2})?$'
        
        if data.get('total_amount'):
            clean_amount = data['total_amount'].replace(' ', '').replace('DH', '').replace('TTC', '')
            if not re.match(amount_pattern, clean_amount):
                errors.append("Invalid total amount format")
        
        # If stamp duty is present, validate it's a number
        if 'stamp_duty' in data and data['stamp_duty'] is not None:
            if not isinstance(data['stamp_duty'], (int, float)):
                errors.append("Stamp duty must be a number")
        
        # Validate date format
        date_pattern = r'\d{2}/\d{2}/\d{4}'
        if data.get('payment_date') and not re.match(date_pattern, data['payment_date']):
            errors.append("Invalid payment date format")
        
        return errors


class BillPaymentReceiptA4Extractor(BillPaymentReceiptExtractor):
    """Extractor specifically for A4 format bill payment receipts"""
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key, is_80mm=False)


class BillPaymentReceipt80mmExtractor(BillPaymentReceiptExtractor):
    """Extractor specifically for 80mm format bill payment receipts"""
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key, is_80mm=True)