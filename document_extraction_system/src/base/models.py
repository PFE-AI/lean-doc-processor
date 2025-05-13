from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


class DocumentType(Enum):
    """Supported document types"""
    ACCOUNT_STATEMENT = "account_statement"
    COMMISSION_STATEMENT = "commission_statement"
    BILL_PAYMENT_RECEIPT_A4 = "bill_payment_receipt_a4"
    BILL_PAYMENT_RECEIPT_80MM = "bill_payment_receipt_80mm"
    GENERIC = "generic"


@dataclass
class ExtractedData:
    """Container for extracted document data"""
    document_type: DocumentType
    raw_text: str
    extracted_fields: Dict[str, Any]
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    ocr_quality_score: Optional[float] = None