# Database-Driven Document Classifier Architecture

## Overview
This design transforms the Moroccan Document Classifier into a fully configurable system where document types, features, keywords, and classification rules can be managed through a database without code changes.

## Key Components

### 1. Database Schema

#### Collections/Tables

**DocumentTypes**
```
{
  "id": "string", // UUID
  "name": "string", // e.g., "payment_bills", "commission_statements", "invoices"
  "description": "string",
  "created_at": "datetime",
  "updated_at": "datetime",
  "active": "boolean"
}
```

**Keywords**
```
{
  "id": "string", // UUID
  "document_type_id": "string", // Foreign key to DocumentTypes
  "term": "string", // e.g., "reçu", "facture"
  "language": "string", // e.g., "fr", "ar"
  "weight": "float", // Importance weight
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

**FeatureDefinitions**
```
{
  "id": "string", // UUID
  "name": "string", // e.g., "amount_count", "has_table"
  "description": "string",
  "extractor_type": "string", // e.g., "regex", "visual", "position"
  "extraction_parameters": "json", // Parameters for the extractor
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

**ClassificationRules**
```
{
  "id": "string", // UUID
  "document_type_id": "string", // Foreign key to DocumentTypes
  "feature_id": "string", // Foreign key to FeatureDefinitions
  "operator": "string", // e.g., ">", "<", "=="
  "threshold": "float",
  "weight": "float", // Importance in classification
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

**ModelConfigurations**
```
{
  "id": "string", // UUID
  "name": "string",
  "model_type": "string", // e.g., "bert", "roberta"
  "pretrained_model_name": "string",
  "hyperparameters": "json", // Learning rate, batch size, etc.
  "feature_importances": "json", // Weight of each feature
  "created_at": "datetime",
  "updated_at": "datetime",
  "active": "boolean"
}
```

**TrainingHistory**
```
{
  "id": "string", // UUID
  "model_config_id": "string", // Foreign key to ModelConfigurations
  "metrics": "json", // Accuracy, F1, etc.
  "train_date": "datetime",
  "version": "string"
}
```

### 2. System Architecture

#### Database Connections Module
```python
class DatabaseConnection:
    """Abstract database connection class."""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        
    def connect(self):
        """Connect to the database."""
        pass
    
    def close(self):
        """Close the database connection."""
        pass
    
    def query(self, query_string, params=None):
        """Execute a query and return results."""
        pass
    
    def insert(self, collection, document):
        """Insert a document into a collection."""
        pass
    
    def update(self, collection, query, update):
        """Update documents in a collection."""
        pass


class MongoDBConnection(DatabaseConnection):
    """MongoDB implementation of DatabaseConnection."""
    
    def connect(self):
        from pymongo import MongoClient
        self.client = MongoClient(self.connection_string)
        self.db = self.client.get_default_database()
        
    def query(self, collection, query=None):
        return list(self.db[collection].find(query or {}))
        
    # Other implementations...


class SQLServerConnection(DatabaseConnection):
    """SQL Server implementation of DatabaseConnection."""
    
    def connect(self):
        import pyodbc
        self.conn = pyodbc.connect(self.connection_string)
        
    def query(self, query_string, params=None):
        cursor = self.conn.cursor()
        cursor.execute(query_string, params or [])
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
        
    # Other implementations...
```

#### Configuration Manager
```python
class ConfigurationManager:
    """Manages all system configurations from the database."""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.document_types = {}
        self.keywords = {}
        self.features = {}
        self.rules = {}
        self.model_config = {}
        self.refresh()
        
    def refresh(self):
        """Reload all configurations from the database."""
        self._load_document_types()
        self._load_keywords()
        self._load_features()
        self._load_rules()
        self._load_model_config()
        
    def _load_document_types(self):
        """Load document types from database."""
        self.document_types = {
            doc["id"]: doc for doc in 
            self.db.query("DocumentTypes", {"active": True})
        }
        
    def _load_keywords(self):
        """Load keywords from database."""
        keywords = self.db.query("Keywords")
        self.keywords = {}
        
        for kw in keywords:
            doc_type_id = kw["document_type_id"]
            language = kw["language"]
            
            if doc_type_id not in self.keywords:
                self.keywords[doc_type_id] = {}
                
            if language not in self.keywords[doc_type_id]:
                self.keywords[doc_type_id][language] = []
                
            self.keywords[doc_type_id][language].append({
                "term": kw["term"],
                "weight": kw["weight"]
            })
    
    # Other loading methods...
    
    def get_document_types(self):
        """Return active document types."""
        return list(self.document_types.values())
        
    def get_keywords_for_document(self, doc_type_id, language):
        """Get keywords for a document type in a specific language."""
        if doc_type_id in self.keywords and language in self.keywords[doc_type_id]:
            return self.keywords[doc_type_id][language]
        return []
        
    # Other accessor methods...
```

#### Dynamic Feature Extractor
```python
class FeatureExtractorFactory:
    """Factory for creating feature extractors based on database configuration."""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.extractors = {}
        self._register_extractors()
        
    def _register_extractors(self):
        """Register all feature extractors."""
        self.extractors = {
            "regex": RegexFeatureExtractor,
            "visual": VisualFeatureExtractor,
            "position": PositionFeatureExtractor,
            "keyword": KeywordFeatureExtractor,
            # Add more extractors as needed
        }
        
    def create_extractor(self, feature_def):
        """Create an extractor based on feature definition."""
        extractor_type = feature_def["extractor_type"]
        if extractor_type not in self.extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
            
        return self.extractors[extractor_type](
            feature_def["name"],
            feature_def["extraction_parameters"]
        )
        
    def create_all_extractors(self):
        """Create all extractors from configuration."""
        extractors = []
        for feature in self.config.get_all_features():
            extractors.append(self.create_extractor(feature))
        return extractors


class BaseFeatureExtractor:
    """Base class for feature extractors."""
    
    def __init__(self, name, params):
        self.name = name
        self.params = params
        
    def extract(self, ocr_data):
        """Extract feature from OCR data."""
        raise NotImplementedError


class RegexFeatureExtractor(BaseFeatureExtractor):
    """Extract features using regular expressions."""
    
    def extract(self, ocr_data):
        import re
        
        pattern = self.params["pattern"]
        text = ' '.join([e['text'] for e in ocr_data['text_elements']])
        
        matches = re.findall(pattern, text)
        return len(matches)


class KeywordFeatureExtractor(BaseFeatureExtractor):
    """Extract features based on keyword presence."""
    
    def extract(self, ocr_data, keywords):
        text = ' '.join([e['text'].lower() for e in ocr_data['text_elements']])
        
        count = 0
        for keyword in keywords:
            if keyword["term"].lower() in text:
                count += keyword["weight"]
                
        return count

# Other feature extractors...
```

#### Dynamic Classification Module
```python
class DynamicClassifier:
    """Classification system that uses database configuration."""
    
    def __init__(self, config_manager, feature_factory, model_loader):
        self.config = config_manager
        self.feature_factory = feature_factory
        self.model_loader = model_loader
        self.extractors = self.feature_factory.create_all_extractors()
        self.model = self.model_loader.load_active_model()
        
    def classify(self, ocr_data):
        """Classify a document based on OCR data."""
        # Extract features
        features = {}
        for extractor in self.extractors:
            features[extractor.name] = extractor.extract(ocr_data)
            
        # Rule-based classification
        rule_scores = self._apply_rules(features)
        
        # Model-based classification
        model_scores = self._apply_model(features)
        
        # Combine scores
        final_scores = self._combine_scores(rule_scores, model_scores)
        
        # Get best match
        best_type_id = max(final_scores, key=final_scores.get)
        confidence = final_scores[best_type_id]
        
        return {
            "document_type_id": best_type_id,
            "document_type_name": self.config.document_types[best_type_id]["name"],
            "confidence": confidence,
            "scores": final_scores
        }
        
    def _apply_rules(self, features):
        """Apply classification rules from database."""
        scores = {doc_type_id: 0 for doc_type_id in self.config.document_types}
        
        for rule in self.config.get_all_rules():
            feature_value = features.get(rule["feature_id"], 0)
            doc_type_id = rule["document_type_id"]
            
            if self._evaluate_rule(rule, feature_value):
                scores[doc_type_id] += rule["weight"]
                
        return scores
        
    def _evaluate_rule(self, rule, value):
        """Evaluate a single classification rule."""
        op = rule["operator"]
        threshold = rule["threshold"]
        
        if op == ">":
            return value > threshold
        elif op == ">=":
            return value >= threshold
        elif op == "<":
            return value < threshold
        elif op == "<=":
            return value <= threshold
        elif op == "==":
            return value == threshold
        else:
            return False
            
    def _apply_model(self, features):
        """Apply ML model for classification."""
        if not self.model:
            return {}
            
        # Convert features to model input format
        model_input = self._prepare_model_input(features)
        
        # Get model predictions
        predictions = self.model.predict(model_input)
        
        # Convert to score format
        scores = {}
        for i, doc_type_id in enumerate(self.config.document_types):
            scores[doc_type_id] = float(predictions[0][i])
            
        return scores
        
    def _combine_scores(self, rule_scores, model_scores):
        """Combine rule-based and model-based scores."""
        combined = {}
        
        # Get weights from configuration
        rule_weight = self.config.model_config.get("rule_weight", 0.5)
        model_weight = 1.0 - rule_weight
        
        for doc_type_id in self.config.document_types:
            rule_score = rule_scores.get(doc_type_id, 0)
            model_score = model_scores.get(doc_type_id, 0)
            
            combined[doc_type_id] = (rule_score * rule_weight) + (model_score * model_weight)
            
        return combined
```

#### Admin API for Configuration Management
```python
class AdminAPI:
    """API for managing document classification configuration."""
    
    def __init__(self, db_connection, config_manager):
        self.db = db_connection
        self.config = config_manager
        
    def add_document_type(self, name, description):
        """Add a new document type."""
        doc_type = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "active": True
        }
        
        self.db.insert("DocumentTypes", doc_type)
        self.config.refresh()
        return doc_type
        
    def add_keyword(self, doc_type_id, term, language, weight=1.0):
        """Add a keyword for a document type."""
        keyword = {
            "id": str(uuid.uuid4()),
            "document_type_id": doc_type_id,
            "term": term,
            "language": language,
            "weight": weight,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        self.db.insert("Keywords", keyword)
        self.config.refresh()
        return keyword
        
    # Other management methods...
```

### 3. Main Application Flow

```python
# Main.py

import os
from configparser import ConfigParser

# Load environment configuration
config = ConfigParser()
config.read('config.ini')

# Initialize database connection
db_type = os.environ.get('DB_TYPE', config.get('Database', 'Type'))
db_connection_string = os.environ.get('DB_CONNECTION', config.get('Database', 'ConnectionString'))

if db_type.lower() == 'mongodb':
    db_connection = MongoDBConnection(db_connection_string)
elif db_type.lower() == 'sqlserver':
    db_connection = SQLServerConnection(db_connection_string)
else:
    raise ValueError(f"Unsupported database type: {db_type}")

db_connection.connect()

# Initialize configuration manager
config_manager = ConfigurationManager(db_connection)

# Initialize feature extractor factory
feature_factory = FeatureExtractorFactory(config_manager)

# Initialize model loader
model_loader = ModelLoader(config_manager)

# Initialize OCR processor
ocr = MoroccanOCR(config_manager)

# Initialize classifier
classifier = DynamicClassifier(config_manager, feature_factory, model_loader)

# Set up API routes
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/classify', methods=['POST'])
def classify_document():
    # Get file from request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    file_path = os.path.join('/tmp', file.filename)
    file.save(file_path)
    
    # Process document
    ocr_data = ocr.extract_text(file_path)
    classification = classifier.classify(ocr_data)
    
    # Clean up
    os.remove(file_path)
    
    return jsonify(classification)

@app.route('/api/admin/document-types', methods=['POST'])
def add_document_type():
    data = request.json
    admin_api = AdminAPI(db_connection, config_manager)
    result = admin_api.add_document_type(data['name'], data['description'])
    return jsonify(result)

# More API routes...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Benefits of This Architecture

1. **Fully Configurable**: All aspects of the system are driven by database configurations
2. **Extensible**: New document types can be added without code changes
3. **Modular**: Clear separation of concerns with factory patterns and dependency injection
4. **Flexible**: Supports multiple database backends (MongoDB, SQL Server, etc.)
5. **Maintainable**: Business logic is isolated from configuration data
6. **Scalable**: Easy to distribute and horizontally scale components

## Implementation Plan

1. **Database Setup**: Create schemas in MongoDB or SQL Server
2. **Configuration Manager**: Implement the core configuration system
3. **Feature Extractors**: Develop the dynamic feature extraction system
4. **Classification Engine**: Build the flexible classification system
5. **Admin API**: Create the management interface
6. **Training System**: Implement the model training infrastructure
7. **Web UI**: Develop an admin interface for configuration

## Technical Considerations

1. **Database Choice**: MongoDB provides flexibility for schema evolution, while SQL Server offers stronger transactional guarantees
2. **Performance Optimization**: Cache configurations in memory and refresh periodically
3. **Security**: Implement role-based access control for admin functions
4. **Monitoring**: Add instrumentation for performance tracking and anomaly detection
5. **Deployment**: Use containerization with Docker for easy deployment









































# Configuration Initialization
# ------------------------------

# config_init.py
"""Initialize and populate the database with document classification configurations."""

import os
import uuid
from datetime import datetime
import argparse
from database_manager import get_db_connection

def init_database(db_conn, sample_data=True):
    """Initialize database with necessary collections/tables."""
    # Create tables/collections if using SQL you'd execute create table statements here
    collections = [
        "DocumentTypes", "Keywords", "FeatureDefinitions", 
        "ClassificationRules", "ModelConfigurations", "TrainingHistory"
    ]
    
    # In MongoDB, collections are created implicitly, but we can validate
    if hasattr(db_conn, 'db'):  # MongoDB connection
        existing_collections = db_conn.db.list_collection_names()
        for collection in collections:
            if collection not in existing_collections:
                print(f"Creating collection: {collection}")
                db_conn.db.create_collection(collection)
    
    # For SQL Server, we would execute CREATE TABLE statements
    if sample_data:
        populate_sample_data(db_conn)

def populate_sample_data(db_conn):
    """Populate database with sample document types and configurations."""
    # Add document types
    document_types = [
        {
            "id": str(uuid.uuid4()),
            "name": "payment_bills",
            "description": "Payment receipts and bills from service providers",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "active": True
        },
        {
            "id": str(uuid.uuid4()),
            "name": "commission_statements",
            "description": "Commission statements for agents and representatives",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "active": True
        },
        {
            "id": str(uuid.uuid4()),
            "name": "invoices",
            "description": "Commercial invoices for products or services",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "active": True
        }
    ]
    
    # Insert document types
    for doc_type in document_types:
        db_conn.insert("DocumentTypes", doc_type)
        print(f"Added document type: {doc_type['name']}")
    
    # Add keywords for each document type
    keywords = []
    
    # Payment bills keywords (French)
    for term in ["reçu", "paiement", "facture", "montant", "dh", "timbre", "inwi"]:
        keywords.append({
            "id": str(uuid.uuid4()),
            "document_type_id": document_types[0]["id"],
            "term": term,
            "language": "fr",
            "weight": 1.0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
    
    # Payment bills keywords (Arabic)
    for term in ["استلام", "دفع", "فاتورة", "مبلغ", "درهم"]:
        keywords.append({
            "id": str(uuid.uuid4()),
            "document_type_id": document_types[0]["id"],
            "term": term,
            "language": "ar",
            "weight": 1.0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
    
    # Commission statements keywords (French)
    for term in ["commission", "solde", "avance", "tva", "n°rc", "patente", "honoraire"]:
        keywords.append({
            "id": str(uuid.uuid4()),
            "document_type_id": document_types[1]["id"],
            "term": term,
            "language": "fr",
            "weight": 1.0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
    
    # Commission statements keywords (Arabic)
    for term in ["عمولة", "رصيد", "مقدم", "ضريبة"]:
        keywords.append({
            "id": str(uuid.uuid4()),
            "document_type_id": document_types[1]["id"],
            "term": term,
            "language": "ar",
            "weight": 1.0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
        
    # Invoice keywords (French)
    for term in ["facture", "client", "numéro", "article", "quantité", "prix", "total"]:
        keywords.append({
            "id": str(uuid.uuid4()),
            "document_type_id": document_types[2]["id"],
            "term": term,
            "language": "fr", 
            "weight": 1.0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
    
    # Insert all keywords
    for keyword in keywords:
        db_conn.insert("Keywords", keyword)
    print(f"Added {len(keywords)} keywords")
    
    # Add feature definitions
    features = [
        {
            "id": str(uuid.uuid4()),
            "name": "amount_count",
            "description": "Count of currency amounts (DH/MAD format)",
            "extractor_type": "regex",
            "extraction_parameters": {
                "pattern": r'\d+[\.,]\d{2}\s*(?:DH|MAD|د\.م\.)'
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": str(uuid.uuid4()),
            "name": "has_table",
            "description": "Presence of table structures",
            "extractor_type": "visual",
            "extraction_parameters": {
                "min_lines": 3,
                "min_columns": 2
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": str(uuid.uuid4()),
            "name": "header_size",
            "description": "Size of the document header",
            "extractor_type": "position",
            "extraction_parameters": {
                "top_percentage": 0.2
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": str(uuid.uuid4()),
            "name": "payment_keywords",
            "description": "Count of payment-related keywords",
            "extractor_type": "keyword",
            "extraction_parameters": {
                "document_type_id": document_types[0]["id"]
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": str(uuid.uuid4()),
            "name": "commission_keywords",
            "description": "Count of commission-related keywords",
            "extractor_type": "keyword",
            "extraction_parameters": {
                "document_type_id": document_types[1]["id"]
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": str(uuid.uuid4()),
            "name": "invoice_keywords",
            "description": "Count of invoice-related keywords",
            "extractor_type": "keyword",
            "extraction_parameters": {
                "document_type_id": document_types[2]["id"]
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
    ]
    
    # Insert feature definitions
    for feature in features:
        db_conn.insert("FeatureDefinitions", feature)
    print(f"Added {len(features)} feature definitions")
    
    # Add classification rules
    rules = [
        # Payment bills rules
        {
            "id": str(uuid.uuid4()),
            "document_type_id": document_types[0]["id"],
            "feature_id": features[0]["id"],  # amount_count
            "operator": ">",
            "threshold": 1.0,
            "weight": 0.7,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": str(uuid.uuid4()),
            "document_type_id": document_types[0]["id"],
            "feature_id": features[3]["id"],  # payment_keywords
            "operator": ">",
            "threshold": 2.0,
            "weight": 1.0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        
        # Commission statements rules
        {
            "id": str(uuid.uuid4()),
            "document_type_id": document_types[1]["id"],
            "feature_id": features[1]["id"],  # has_table
            "operator": "==",
            "threshold": 1.0,
            "weight": 0.8,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": str(uuid.uuid4()),
            "document_type_id": document_types[1]["id"],
            "feature_id": features[4]["id"],  # commission_keywords
            "operator": ">",
            "threshold": 1.0,
            "weight": 1.0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        
        # Invoice rules
        {
            "id": str(uuid.uuid4()),
            "document_type_id": document_types[2]["id"],
            "feature_id": features[5]["id"],  # invoice_keywords
            "operator": ">",
            "threshold": 3.0,
            "weight": 1.0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": str(uuid.uuid4()),
            "document_type_id": document_types[2]["id"],
            "feature_id": features[1]["id"],  # has_table
            "operator": "==",
            "threshold": 1.0,
            "weight": 0.6,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
    ]
    
    # Insert classification rules
    for rule in rules:
        db_conn.insert("ClassificationRules", rule)
    print(f"Added {len(rules)} classification rules")
    
    # Add model configuration
    model_config = {
        "id": str(uuid.uuid4()),
        "name": "BERT with features model",
        "model_type": "bert",
        "pretrained_model_name": "bert-base-multilingual-cased",
        "hyperparameters": {
            "learning_rate": 2e-5,
            "batch_size": 8,
            "num_epochs": 3,
            "max_length": 512
        },
        "feature_importances": {
            "text_features": 0.7,
            "extracted_features": 0.3,
            "rule_weight": 0.4
        },
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "active": True
    }
    
    db_conn.insert("ModelConfigurations", model_config)
    print("Added model configuration")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize document classifier database")
    parser.add_argument("--sample", action="store_true", help="Add sample data")
    args = parser.parse_args()
    
    db_conn = get_db_connection()
    init_database(db_conn, sample_data=args.sample)
    print("Database initialization complete")


# Database Connection Managers
# ------------------------------

# database_manager.py
"""Database connection managers for different database backends."""

import os
import json
from configparser import ConfigParser

def get_db_connection():
    """Factory function to get the appropriate database connection."""
    # Read configuration
    config = ConfigParser()
    config.read('config.ini')
    
    db_type = os.environ.get('DB_TYPE', config.get('Database', 'Type', fallback='mongodb'))
    connection_string = os.environ.get('DB_CONNECTION', 
                                       config.get('Database', 'ConnectionString', 
                                                 fallback='mongodb://localhost:27017/doc_classifier'))
    
    # Create appropriate connection
    if db_type.lower() == 'mongodb':
        return MongoDBConnection(connection_string)
    elif db_type.lower() == 'sqlserver':
        return SQLServerConnection(connection_string)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


class DatabaseConnection:
    """Abstract database connection class."""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        
    def connect(self):
        """Connect to the database."""
        raise NotImplementedError
    
    def close(self):
        """Close the database connection."""
        raise NotImplementedError
    
    def query(self, collection, query=None):
        """Query documents from a collection."""
        raise NotImplementedError
    
    def insert(self, collection, document):
        """Insert a document into a collection."""
        raise NotImplementedError
    
    def update(self, collection, query, update):
        """Update documents in a collection."""
        raise NotImplementedError
    
    def delete(self, collection, query):
        """Delete documents from a collection."""
        raise NotImplementedError


class MongoDBConnection(DatabaseConnection):
    """MongoDB implementation of DatabaseConnection."""
    
    def connect(self):
        """Connect to MongoDB."""
        from pymongo import MongoClient
        
        self.client = MongoClient(self.connection_string)
        self.db = self.client.get_default_database()
        return self
    
    def close(self):
        """Close the MongoDB connection."""
        if hasattr(self, 'client'):
            self.client.close()
    
    def query(self, collection, query=None):
        """Query documents from a MongoDB collection."""
        if not hasattr(self, 'db'):
            self.connect()
        
        return list(self.db[collection].find(query or {}))
    
    def insert(self, collection, document):
        """Insert a document into a MongoDB collection."""
        if not hasattr(self, 'db'):
            self.connect()
        
        result = self.db[collection].insert_one(document)
        return str(result.inserted_id)
    
    def update(self, collection, query, update):
        """Update documents in a MongoDB collection."""
        if not hasattr(self, 'db'):
            self.connect()
        
        result = self.db[collection].update_many(query, {"$set": update})
        return result.modified_count
    
    def delete(self, collection, query):
        """Delete documents from a MongoDB collection."""
        if not hasattr(self, 'db'):
            self.connect()
        
        result = self.db[collection].delete_many(query)
        return result.deleted_count


class SQLServerConnection(DatabaseConnection):
    """SQL Server implementation of DatabaseConnection."""
    
    def connect(self):
        """Connect to SQL Server."""
        import pyodbc
        
        self.conn = pyodbc.connect(self.connection_string)
        self.conn.autocommit = False
        return self
    
    def close(self):
        """Close the SQL Server connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def query(self, table, query=None):
        """Query records from a SQL Server table."""
        if not hasattr(self, 'conn'):
            self.connect()
        
        cursor = self.conn.cursor()
        
        # Build query based on filter conditions
        if isinstance(query, dict) and query:
            conditions = []
            params = []
            
            for key, value in query.items():
                conditions.append(f"{key} = ?")
                params.append(value)
            
            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            sql = f"SELECT * FROM {table}{where_clause}"
            cursor.execute(sql, params)
        else:
            cursor.execute(f"SELECT * FROM {table}")
        
        # Convert to dictionaries
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        cursor.close()
        return results
    
    def insert(self, table, document):
        """Insert a record into a SQL Server table."""
        if not hasattr(self, 'conn'):
            self.connect()
        
        cursor = self.conn.cursor()
        
        # Prepare columns and values
        columns = list(document.keys())
        placeholders = ["?"] * len(columns)
        values = [document[col] for col in columns]
        
        # Handle JSON/dict values
        for i, value in enumerate(values):
            if isinstance(value, (dict, list)):
                values[i] = json.dumps(value)
        
        # Execute insert
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        cursor.execute(sql, values)
        self.conn.commit()
        
        # Return the ID
        return document.get('id', None)
    
    def update(self, table, query, update):
        """Update records in a SQL Server table."""
        if not hasattr(self, 'conn'):
            self.connect()
        
        cursor = self.conn.cursor()
        
        # Build update statement
        set_clause = []
        params = []
        
        for key, value in update.items():
            set_clause.append(f"{key} = ?")
            
            # Handle JSON/dict values
            if isinstance(value, (dict, list)):
                params.append(json.dumps(value))
            else:
                params.append(value)
        
        # Build where clause
        conditions = []
        for key, value in query.items():
            conditions.append(f"{key} = ?")
            params.append(value)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Execute update
        sql = f"UPDATE {table} SET {', '.join(set_clause)}{where_clause}"
        cursor.execute(sql, params)
        count = cursor.rowcount
        self.conn.commit()
        
        return count
    
    def delete(self, table, query):
        """Delete records from a SQL Server table."""
        if not hasattr(self, 'conn'):
            self.connect()
        
        cursor = self.conn.cursor()
        
        # Build where clause
        conditions = []
        params = []
        
        for key, value in query.items():
            conditions.append(f"{key} = ?")
            params.append(value)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Execute delete
        sql = f"DELETE FROM {table}{where_clause}"
        cursor.execute(sql, params)
        count = cursor.rowcount
        self.conn.commit()
        
        return count


# Configuration Manager
# ------------------------------

# config_manager.py
"""Configuration manager for loading document classification settings from database."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

class ConfigurationManager:
    """Manages all system configurations from the database."""
    
    def __init__(self, db_connection):
        """
        Initialize configuration manager.
        
        Args:
            db_connection: Database connection object
        """
        self.db = db_connection
        self.document_types = {}
        self.keywords = {}
        self.features = {}
        self.rules = {}
        self.model_config = {}
        self.last_refresh = None
        self.refresh_interval = 300  # 5 minutes refresh interval
        
        # Initialize logger
        self.logger = logging.getLogger("ConfigurationManager")
        
        # Load initial configurations
        self.refresh()
    
    def refresh(self):
        """Reload all configurations from the database."""
        self.logger.info("Refreshing configurations from database")
        self._load_document_types()
        self._load_keywords()
        self._load_features()
        self._load_rules()
        self._load_model_config()
        self.last_refresh = datetime.now()
        self.logger.info("Configuration refresh complete")
    
    def check_refresh(self):
        """Check if configurations need to be refreshed."""
        if self.last_refresh is None:
            return self.refresh()
        
        time_elapsed = (datetime.now() - self.last_refresh).total_seconds()
        if time_elapsed > self.refresh_interval:
            return self.refresh()
    
    def _load_document_types(self):
        """Load document types from database."""
        try:
            docs = self.db.query("DocumentTypes", {"active": True})
            self.document_types = {doc["id"]: doc for doc in docs}
            self.logger.info(f"Loaded {len(self.document_types)} document types")
        except Exception as e:
            self.logger.error(f"Error loading document types: {str(e)}")
    
    def _load_keywords(self):
        """Load keywords from database."""
        try:
            keywords = self.db.query("Keywords")
            self.keywords = {}
            
            for kw in keywords:
                doc_type_id = kw["document_type_id"]
                language = kw["language"]
                
                if doc_type_id not in self.keywords:
                    self.keywords[doc_type_id] = {}
                    
                if language not in self.keywords[doc_type_id]:
                    self.keywords[doc_type_id][language] = []
                    
                self.keywords[doc_type_id][language].append({
                    "term": kw["term"],
                    "weight": kw["weight"]
                })
            
            self.logger.info(f"Loaded keywords for {len(self.keywords)} document types")
        except Exception as e:
            self.logger.error(f"Error loading keywords: {str(e)}")
    
    def _load_features(self):
        """Load feature definitions from database."""
        try:
            features = self.db.query("FeatureDefinitions")
            self.features = {feature["id"]: feature for feature in features}
            self.logger.info(f"Loaded {len(self.features)} feature definitions")
        except Exception as e:
            self.logger.error(f"Error loading features: {str(e)}")
    
    def _load_rules(self):
        """Load classification rules from database."""
        try:
            rules = self.db.query("ClassificationRules")
            self.rules = {}
            
            for rule in rules:
                doc_type_id = rule["document_type_id"]
                
                if doc_type_id not in self.rules:
                    self.rules[doc_type_id] = []
                
                self.rules[doc_type_id].append(rule)
            
            self.logger.info(f"Loaded rules for {len(self.rules)} document types")
        except Exception as e:
            self.logger.error(f"Error loading rules: {str(e)}")
    
    def _load_model_config(self):
        """Load active model configuration."""
        try:
            configs = self.db.query("ModelConfigurations", {"active": True})
            if configs:
                self.model_config = configs[0]
                self.logger.info(f"Loaded active model configuration: {self.model_config['name']}")
            else:
                self.logger.warning("No active model configuration found")
        except Exception as e:
            self.logger.error(f"Error loading model configuration: {str(e)}")
    
    def get_document_types(self) -> List[Dict[str, Any]]:
        """Return active document types."""
        self.check_refresh()
        return list(self.document_types.values())
    
    def get_document_type(self, doc_type_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document type by ID."""
        self.check_refresh()
        return self.document_types.get(doc_type_id)
    
    def get_keywords_for_document(self, doc_type_id: str, language: str = None) -> List[Dict[str, Any]]:
        """Get keywords for a document type in a specific language."""
        self.check_refresh()
        
        if doc_type_id not in self.keywords:
            return []
        
        if language and language in self.keywords[doc_type_id]:
            return self.keywords[doc_type_id][language]
        
        # If no language specified, return all keywords
        all_keywords = []
        for lang_keywords in self.keywords[doc_type_id].values():
            all_keywords.extend(lang_keywords)
        
        return all_keywords
    
    def get_all_features(self) -> List[Dict[str, Any]]:
        """Get all feature definitions."""
        self.check_refresh()
        return list(self.features.values())
    
    def get_feature(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific feature by ID."""
        self.check_refresh()
        return self.features.get(feature_id)
    
    def get_rules_for_document(self, doc_type_id: str) -> List[Dict[str, Any]]:
        """Get classification rules for a document type."""
        self.check_refresh()
        return self.rules.get(doc_type_id, [])
    
    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Get all classification rules."""
        self.check_refresh()
        all_rules = []
        for rules in self.rules.values():
            all_rules.extend(rules)
        return all_rules
    
    def get_active_model_config(self) -> Optional[Dict[str, Any]]:
        """Get the active model configuration."""
        self.check_refresh()
        return self.model_config if self.model_config else None



















        