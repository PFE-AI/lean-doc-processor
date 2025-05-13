# README.md
# Document Extraction System

A comprehensive system for extracting structured information from various types of documents using Mistral OCR and LangChain.

## Features

- **OCR Processing**: Utilizes Mistral's OCR capabilities for text extraction
- **Multiple Document Types**: Supports Account Statements, Commission Statements, and Bill Payment Receipts (A4 and 80mm formats)
- **Auto-Detection**: Automatically detects document type based on content
- **Structured Extraction**: Uses Pydantic models for structured data extraction
- **Validation**: Built-in validation for extracted data
- **Batch Processing**: Process multiple documents at once
- **Extensible**: Easy to add new document types

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-extraction-system.git
cd document-extraction-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export MISTRAL_API_KEY="your-mistral-api-key"
```

Or create a `.env` file:
```
MISTRAL_API_KEY=your-mistral-api-key
```

## Usage

### Basic Usage

```python
from src.pipeline import DocumentExtractionPipeline
from src.base.models import DocumentType

# Initialize pipeline
pipeline = DocumentExtractionPipeline()

# Extract from a document with auto-detection
result = pipeline.extract_from_document("path/to/document.pdf")

# Extract with specific document type
result = pipeline.extract_from_document(
    "path/to/receipt.pdf",
    document_type=DocumentType.BILL_PAYMENT_RECEIPT_A4
)

# Access extracted data
print(result.extracted_fields)
print(result.validation_errors)
```

### Batch Processing

```python
documents = [
    {"path": "doc1.pdf", "type": DocumentType.ACCOUNT_STATEMENT},
    {"path": "doc2.pdf"},  # Auto-detect type
    {"path": "doc3.pdf", "type": DocumentType.COMMISSION_STATEMENT}
]

results = pipeline.batch_extract(documents)
```

### Validation

```python
# Get validation report
report = pipeline.validate_extraction(result)
print(report['is_valid'])
print(report['field_completeness'])
```

## Supported Document Types

1. **Account Statement** (`account_statement`)
   - Bank account statements
   - Extracts account holder info, balances, transactions

2. **Commission Statement** (`commission_statement`)
   - Agent commission statements
   - Extracts agent info, commission details, totals

3. **Bill Payment Receipt A4** (`bill_payment_receipt_a4`)
   - Standard A4 format receipts
   - Extracts payment info, bill details, totals

4. **Bill Payment Receipt 80mm** (`bill_payment_receipt_80mm`)
   - Thermal printer format receipts
   - Same fields as A4 but optimized for 80mm format

## Adding New Document Types

1. Create a new extractor in `src/extractors/`:
```python
from src.base.extractor import BaseDocumentExtractor
from src.base.models import DocumentType

class NewDocumentExtractor(BaseDocumentExtractor):
    def __init__(self, api_key=None):
        super().__init__(DocumentType.NEW_TYPE, api_key)
    
    def create_output_model(self):
        # Define Pydantic model
        pass
    
    def create_extraction_prompt(self):
        # Define extraction prompt
        pass
    
    def validate_extracted_data(self, data):
        # Implement validation logic
        pass
```

2. Register the extractor in `src/pipeline.py`:
```python
self.factory.register_extractor(
    DocumentType.NEW_TYPE,
    NewDocumentExtractor
)
```

3. Add configuration in `src/config/document_templates.yaml`

## Project Structure

```
document_extraction_system/
├── src/
│   ├── base/          # Base classes and models
│   ├── extractors/    # Document-specific extractors
│   ├── processors/    # OCR processor
│   ├── config/        # Configuration files
│   └── pipeline.py    # Main pipeline
├── examples/          # Usage examples
├── tests/            # Test suite
└── requirements.txt  # Dependencies
```

## Testing

Run tests with:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.



document_extraction_system/
├── src/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── extractor.py
│   │   └── factory.py
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── account_statement.py
│   │   ├── commission_statement.py
│   │   └── bill_payment_receipt.py
│   ├── processors/
│   │   ├── __init__.py
│   │   └── mistral_ocr.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── document_templates.yaml
│   └── pipeline.py
├── tests/
│   ├── __init__.py
│   └── test_extractors.py
├── examples/
│   └── example_usage.py
├── requirements.txt
└── README.md