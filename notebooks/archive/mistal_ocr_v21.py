from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from .base_extraction_system import (
    BaseDocumentExtractor,
    DocumentType,
    DocumentExtractorFactory
)

# Define document types specifically for the templates
DocumentType.ACCOUNT_STATEMENT = "account_statement"
DocumentType.COMMISSION_STATEMENT = "commission_statement"
DocumentType.BILL_PAYMENT_RECEIPT = "bill_payment_receipt"


# Account Statement Extractor (based on iAccountStatementReport)
class AccountStatementExtractor(BaseDocumentExtractor):
    """Extractor for account statements (relevé de compte)"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(DocumentType.ACCOUNT_STATEMENT, api_key)
    
    def create_output_model(self) -> Type[BaseModel]:
        """Create Pydantic model for account statement"""
        
        class OperationSummary(BaseModel):
            operation_day: str = Field(description="Date Opération / تاريخ العملية")
            operations_count: int = Field(description="Nombre d'opérations / عدد العمليات")
            description: str = Field(description="Description / وصف العملية")
            service_name: str = Field(description="Nom du service")
            debit_amount: float = Field(description="Au débit / في مدينية الحساب")
            credit_amount: float = Field(description="Au crédit / في دائنية الحساب")
            
        class AccountStatement(BaseModel):
            # Account holder information
            account_holder_name: str = Field(description="Nom du titulaire (customer name)")
            account_holder_address: str = Field(description="Adresse du titulaire")
            
            # Account information
            account_number: str = Field(description="RIB / رقم الحساب")
            currency: str = Field(description="Devise / العملة")
            
            # Statement period
            start_period: str = Field(description="Date de début de période")
            end_period: str = Field(description="Date de fin de période")
            
            # Balances
            opening_balance: float = Field(description="Solde de départ / رصيد الحساب في")
            closing_balance: float = Field(description="Nouveau Solde / الرصيد الجديد")
            total_debits: float = Field(description="Total des débits / إجمالي المدينية")
            total_credits: float = Field(description="Total des crédits / إجمالي الدائنية")
            
            # Operations
            operations: List[OperationSummary] = Field(description="Liste des opérations")
            
            # Additional information
            message: Optional[str] = Field(None, description="Message")
            
        return AccountStatement
    
    def create_extraction_prompt(self) -> PromptTemplate:
        """Create prompt for account statement extraction"""
        template = """
        Tu es un expert en extraction d'information à partir de relevés de compte bancaires.
        Le document contient des informations en français et en arabe.
        
        Extrait les informations suivantes du relevé de compte :

        - Nom du titulaire du compte (customer name)
        - Adresse du titulaire 
        - Numéro de compte (RIB)
        - Devise (currency)
        - Période du relevé (date de début et de fin)
        - Solde d'ouverture (opening balance)
        - Solde de clôture (closing balance)
        - Total des débits
        - Total des crédits
        - Liste des opérations avec :
          - Date de l'opération
          - Nombre d'opérations
          - Description
          - Montant au débit
          - Montant au crédit
        - Message (s'il existe)

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
            'account_holder_name', 'account_number', 'start_period', 'end_period',
            'opening_balance', 'closing_balance', 'total_debits', 'total_credits'
        ]
        
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate that operations list exists
        if not data.get('operations'):
            errors.append("No operations found in the statement")
        
        # Validate balance consistency
        if 'opening_balance' in data and 'closing_balance' in data:
            if 'operations' in data and data['operations']:
                total_debits = sum(op.get('debit_amount', 0) for op in data['operations'])
                total_credits = sum(op.get('credit_amount', 0) for op in data['operations'])
                
                # Check if totals match
                if abs(total_debits - data.get('total_debits', 0)) > 0.01:
                    errors.append("Sum of debit operations doesn't match total debits")
                
                if abs(total_credits - data.get('total_credits', 0)) > 0.01:
                    errors.append("Sum of credit operations doesn't match total credits")
        
        return errors


# Commission Statement Extractor
class CommissionStatementExtractor(BaseDocumentExtractor):
    """Extractor for commission statements"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(DocumentType.COMMISSION_STATEMENT, api_key)
    
    def create_output_model(self) -> Type[BaseModel]:
        """Create Pydantic model for commission statement"""
        
        class CommissionDetail(BaseModel):
            service_type: str = Field(description="Type de service")
            transaction_count: int = Field(description="Nombre de transactions")
            commission_rate: float = Field(description="Taux de commission (%)")
            commission_amount: float = Field(description="Montant de commission")
            
        class CommissionStatement(BaseModel):
            # Agent information
            agent_code: str = Field(description="Code agent")
            agent_name: str = Field(description="Nom de l'agent")
            agency_name: Optional[str] = Field(None, description="Nom de l'agence")
            
            # Statement period
            period: str = Field(description="Période de commission")
            statement_date: str = Field(description="Date du relevé")
            
            # Commission details
            total_transactions: int = Field(description="Nombre total de transactions")
            total_transaction_amount: float = Field(description="Montant total des transactions")
            
            # Commission breakdown
            commission_details: List[CommissionDetail] = Field(description="Détails des commissions")
            
            # Summary
            gross_commission: float = Field(description="Commission brute")
            deductions: Optional[float] = Field(None, description="Déductions")
            net_commission: float = Field(description="Commission nette")
            
            # Payment information
            payment_method: Optional[str] = Field(None, description="Mode de paiement")
            payment_date: Optional[str] = Field(None, description="Date de paiement")
            
        return CommissionStatement
    
    def create_extraction_prompt(self) -> PromptTemplate:
        """Create prompt for commission statement extraction"""
        template = """
        Tu es un expert en extraction d'information à partir de relevés de commissions.
        
        Extrait les informations suivantes du relevé de commission :

        - Code agent
        - Nom de l'agent
        - Nom de l'agence (si disponible)
        - Période de commission
        - Date du relevé
        - Nombre total de transactions
        - Montant total des transactions
        - Détails des commissions par type de service :
          - Type de service
          - Nombre de transactions
          - Taux de commission
          - Montant de commission
        - Commission brute totale
        - Déductions (si applicable)
        - Commission nette
        - Mode de paiement (si indiqué)
        - Date de paiement (si indiquée)

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
        """Validate commission statement data"""
        errors = []
        
        # Check required fields
        required_fields = [
            'agent_code', 'agent_name', 'period', 'total_transactions',
            'gross_commission', 'net_commission'
        ]
        
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate commission calculations
        if 'commission_details' in data and data['commission_details']:
            calculated_commission = sum(detail.get('commission_amount', 0) 
                                      for detail in data['commission_details'])
            
            if abs(calculated_commission - data.get('gross_commission', 0)) > 0.01:
                errors.append("Sum of commission details doesn't match gross commission")
        
        # Validate net commission calculation
        if all(field in data for field in ['gross_commission', 'net_commission']):
            if 'deductions' in data and data['deductions']:
                expected_net = data['gross_commission'] - data['deductions']
                if abs(expected_net - data['net_commission']) > 0.01:
                    errors.append("Net commission calculation doesn't match (gross - deductions)")
            else:
                if abs(data['gross_commission'] - data['net_commission']) > 0.01:
                    errors.append("Net commission should equal gross commission when no deductions")
        
        return errors


# Bill Payment Receipt Extractor
class BillPaymentReceiptExtractor(BaseDocumentExtractor):
    """Extractor for bill payment receipts (reçu de paiement)"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(DocumentType.BILL_PAYMENT_RECEIPT, api_key)
    
    def create_output_model(self) -> Type[BaseModel]:
        """Create Pydantic model for bill payment receipt"""
        
        class PaidBill(BaseModel):
            reference: str = Field(description="Référence")
            description: str = Field(description="Description")
            amount: str = Field(description="Montant (DH TTC)")
            
        class AdditionalInfo(BaseModel):
            key: str = Field(description="Nom du champ")
            value: str = Field(description="Valeur du champ")
            
        class BillPaymentReceipt(BaseModel):
            # Receipt information
            payment_date: str = Field(description="Date de paiement")
            receipt_number: str = Field(description="N° reçu")
            creditor_tx_number: str = Field(description="N° Tx créancier")
            inwi_tx_number: str = Field(description="N° Tx inwi money")
            agent_code: str = Field(description="Code agent")
            
            # Optional reference
            reference: Optional[str] = Field(None, description="Référence")
            immatriculation: Optional[str] = Field(None, description="Immatriculation")
            
            # Merchant information
            merchant_name: str = Field(description="Nom du marchand")
            
            # Paid bills
            paid_bills: List[PaidBill] = Field(description="Factures payées")
            
            # Fees and taxes
            management_fees: str = Field(description="Frais de gestion")
            stamp_duty: Optional[float] = Field(None, description="Droits de timbre")
            total_amount: str = Field(description="Total (DH TTC)")
            
            # Additional information
            additional_info: Optional[List[AdditionalInfo]] = Field(None, description="Informations supplémentaires")
            
            # Marketing messages
            user_field2: Optional[str] = Field(None, description="Message utilisateur 2")
            marketing_message: Optional[str] = Field(None, description="Message marketing")
            
        return BillPaymentReceipt
    
    def create_extraction_prompt(self) -> PromptTemplate:
        """Create prompt for bill payment receipt extraction"""
        template = """
        Tu es un expert en extraction d'information à partir de reçus de paiement de factures.
        
        Extrait les informations suivantes du reçu de paiement :

        - Date de paiement
        - N° reçu
        - N° Tx créancier
        - N° Tx inwi money  
        - Code agent
        - Référence (si disponible)
        - Immatriculation (si disponible)
        - Nom du marchand
        - Liste des factures payées avec :
          - Référence
          - Description
          - Montant (DH TTC)
        - Frais de gestion
        - Droits de timbre (si applicable)
        - Total (DH TTC)
        - Informations supplémentaires (si disponibles)
        - Messages marketing (si disponibles)

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
        """Validate bill payment receipt data"""
        errors = []
        
        # Check required fields
        required_fields = [
            'payment_date', 'receipt_number', 'creditor_tx_number',
            'inwi_tx_number', 'agent_code', 'merchant_name', 'total_amount',
            'management_fees'
        ]
        
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate that at least one bill is present
        if not data.get('paid_bills') or len(data['paid_bills']) == 0:
            errors.append("At least one paid bill must be present")
        
        # Validate bill structure
        if 'paid_bills' in data:
            for i, bill in enumerate(data['paid_bills']):
                if not bill.get('reference'):
                    errors.append(f"Bill {i+1}: missing reference")
                if not bill.get('amount'):
                    errors.append(f"Bill {i+1}: missing amount")
        
        return errors


# Register all extractors
DocumentExtractorFactory.register_extractor(
    DocumentType.ACCOUNT_STATEMENT,
    AccountStatementExtractor
)

DocumentExtractorFactory.register_extractor(
    DocumentType.COMMISSION_STATEMENT,
    CommissionStatementExtractor
)

DocumentExtractorFactory.register_extractor(
    DocumentType.BILL_PAYMENT_RECEIPT,
    BillPaymentReceiptExtractor
)


# Example usage specific to these document types
def example_usage():
    """Example usage of the document extractors"""
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
    print(f"Account Holder: {result.extracted_fields.get('account_holder_name')}")
    print(f"Account Number: {result.extracted_fields.get('account_number')}")
    print(f"Opening Balance: {result.extracted_fields.get('opening_balance')}")
    print(f"Closing Balance: {result.extracted_fields.get('closing_balance')}")
    print(f"Number of Operations: {len(result.extracted_fields.get('operations', []))}")
    
    # Extract from bill payment receipt
    receipt_path = "path/to/bill_payment_receipt.pdf"
    result = pipeline.extract_from_document(
        receipt_path,
        DocumentType.BILL_PAYMENT_RECEIPT,
        use_structured_output=True
    )
    
    print("\nBill Payment Receipt Extraction:")
    print(f"Receipt Number: {result.extracted_fields.get('receipt_number')}")
    print(f"Payment Date: {result.extracted_fields.get('payment_date')}")
    print(f"Merchant: {result.extracted_fields.get('merchant_name')}")
    print(f"Total Amount: {result.extracted_fields.get('total_amount')}")
    print(f"Number of Bills: {len(result.extracted_fields.get('paid_bills', []))}")
    
    # Extract from commission statement
    commission_path = "path/to/commission_statement.pdf"
    result = pipeline.extract_from_document(
        commission_path,
        DocumentType.COMMISSION_STATEMENT,
        use_structured_output=True
    )
    
    print("\nCommission Statement Extraction:")
    print(f"Agent Code: {result.extracted_fields.get('agent_code')}")
    print(f"Agent Name: {result.extracted_fields.get('agent_name')}")
    print(f"Period: {result.extracted_fields.get('period')}")
    print(f"Gross Commission: {result.extracted_fields.get('gross_commission')}")
    print(f"Net Commission: {result.extracted_fields.get('net_commission')}")


if __name__ == "__main__":
    example_usage()