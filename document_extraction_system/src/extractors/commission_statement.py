# src/extractors/commission_statement.py
from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from ..base.extractor import BaseDocumentExtractor
from ..base.models import DocumentType


class CommissionStatementExtractor(BaseDocumentExtractor):
    """Extractor for commission statements"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(DocumentType.COMMISSION_STATEMENT, api_key)
    
    def create_output_model(self) -> Type[BaseModel]:
        """Create Pydantic model for commission statement"""
        class CommissionDetail(BaseModel):
            service_name: str = Field(description="Nom du service")
            service_code: Optional[str] = Field(None, description="Code du service")
            transaction_count: int = Field(description="Nombre de transactions")
            total_amount: float = Field(description="Montant total")
            commission_rate: Optional[float] = Field(None, description="Taux de commission")
            commission_amount: float = Field(description="Montant de commission")
            
        class CommissionStatement(BaseModel):
            period: str = Field(description="Période de commission")
            agent_code: str = Field(description="Code agent")
            agent_name: str = Field(description="Nom de l'agent")
            agent_type: Optional[str] = Field(None, description="Type d'agent")
            total_transactions: int = Field(description="Nombre total de transactions")
            total_amount: float = Field(description="Montant total des transactions")
            commission_details: List[CommissionDetail] = Field(description="Détails par service")
            total_commission: float = Field(description="Commission totale")
            deductions: Optional[float] = Field(None, description="Déductions")
            net_commission: float = Field(description="Commission nette")
            payment_date: Optional[str] = Field(None, description="Date de paiement")
            
        return CommissionStatement
    
    def create_extraction_prompt(self) -> PromptTemplate:
        """Create prompt for commission statement extraction"""
        template = """
        You are an expert at extracting information from commission statements.
        Extract the following information from the document text:

        - Commission period
        - Agent code
        - Agent name
        - Agent type (if mentioned)
        - Total number of transactions
        - Total transaction amount
        - Commission details by service (if available):
          - Service name
          - Service code
          - Number of transactions
          - Total amount
          - Commission rate
          - Commission amount
        - Total commission
        - Deductions (if any)
        - Net commission
        - Payment date (if mentioned)

        The document is likely related to inwi money services and may contain various service types.
        
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
            'period', 'agent_code', 'agent_name', 'total_transactions',
            'total_amount', 'total_commission', 'net_commission'
        ]
        
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate numeric fields
        if data.get('total_transactions', 0) <= 0:
            errors.append("Total transactions must be positive")
        
        if data.get('total_amount', 0) <= 0:
            errors.append("Total amount must be positive")
        
        # Validate commission calculations
        if 'commission_details' in data and data['commission_details']:
            calculated_commission = sum(detail.get('commission_amount', 0) 
                                      for detail in data['commission_details'])
            
            # Check if calculated commission matches total commission
            if abs(calculated_commission - data.get('total_commission', 0)) > 0.01:
                errors.append("Sum of commission details doesn't match total commission")
        
        # Validate net commission calculation
        if 'deductions' in data and data['deductions'] is not None:
            calculated_net = data.get('total_commission', 0) - data.get('deductions', 0)
            if abs(calculated_net - data.get('net_commission', 0)) > 0.01:
                errors.append("Net commission calculation doesn't match (total - deductions)")
        
        # Validate commission rates if present
        if 'commission_details' in data:
            for detail in data['commission_details']:
                if 'commission_rate' in detail and detail['commission_rate'] is not None:
                    if detail['commission_rate'] < 0 or detail['commission_rate'] > 100:
                        errors.append(f"Invalid commission rate for {detail.get('service_name', 'unknown service')}")
        
        return errors