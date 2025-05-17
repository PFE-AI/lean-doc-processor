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
            transaction_count: int = Field(description="Nombre de transactions")
            amount: float = Field(description="Montant (en DH HT)")
            vat_amount: Optional[float] = Field(None, description="TVA (20%) en DH")
            net_amount: float = Field(description="Montant (en DH TTC)")
            
        class CommissionStatement(BaseModel):
            period: str = Field(description="Période de commission")
            invoice_reference: str = Field(description="Reference N°")
            agent_code: str = Field(description="Code de l'agent")
            address: str = Field(description="Adresse de l'agent")
            phone_number: str = Field(description="Numéro de téléphone de l'agent")
            total_transactions: int = Field(description="Nombre total de transactions")
            commission_details: List[CommissionDetail] = Field(description="Détails par service")
            total_commission: float = Field(description="Commission totale")
            total_vat: float = Field(description="TVA totale")
            net_total_commission_paid: float = Field(description="Avance su commissions (en DH TTC)")
            net_total_commission_remain: float = Field(description="Net à payer des commissions (en DH TTC)")
            sum: Optional[str] = Field(None, description="La somme de l'arrête des commission")
            
        return CommissionStatement
    
    def create_extraction_prompt(self) -> PromptTemplate:
        """Create prompt for commission statement extraction"""
        template = """
        You are an expert at extracting information from commission statements.
        Extract the following information from the document text:

        - Numero
        - Date
        - Adresse
        - Téléphone
        - Objet
        - Commission details by service (if available):
          - Désignation
          - Quantité
          - Prix (en DH HT)
          - TVA (20%) en DH
          - Prix (en DH TTC)
        - Total (HT)
        - TVA (20%) en DH
        - Avance su commissions (en DH TTC)
        - Net à payer (en DH TTC)
        - La présente facture est arrêtée à la somme de :

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
        
        return errors