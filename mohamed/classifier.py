# OCR Processing Module
# ------------------------------

import os
import sys
import cv2
import numpy as np
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from PIL import Image
import pytesseract
from pytesseract import Output

class MoroccanOCR:
    """OCR processor optimized for Moroccan documents with French and Arabic text."""
    
    def __init__(self, config_manager=None):
        """
        Initialize the OCR processor.
        
        Args:
            config_manager: Configuration manager
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger("MoroccanOCR")
        
        # Load configuration from database if available, otherwise use defaults
        tesseract_config = None
        ocr_confidence_threshold = 30
        tesseract_path = "/usr/bin/tesseract"
        
        if config_manager:
            model_config = config_manager.get_active_model_config() or {}
            ocr_settings = model_config.get("ocr_settings", {})
            tesseract_config = ocr_settings.get("tesseract_config", "--psm 6 -l fra+ara")
            ocr_confidence_threshold = ocr_settings.get("confidence_threshold", 30)
            tesseract_path = ocr_settings.get("tesseract_path", "/usr/bin/tesseract")
        
        # Initialize OCR settings
        self.tesseract_config = tesseract_config or "--psm 6 -l fra+ara"
        self.confidence_threshold = ocr_confidence_threshold
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.logger.info("OCR processor initialized", 
                        extra={"tesseract_path": tesseract_path, 
                              "config": self.tesseract_config})
    
    def preprocess(self, img_path: str) -> Optional[np.ndarray]:
        """
        Preprocess an image for OCR.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array, or None if preprocessing failed
        """
        start_time = time.time()
        try:
            # Input validation
            if not os.path.exists(img_path):
                self.logger.error(f"Image file not found: {img_path}")
                return None
                
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                self.logger.error(f"Could not read image: {img_path}")
                return None

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Thresholding
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Log processing time
            self.logger.info(f"Image preprocessing completed in {time.time() - start_time:.2f}s", 
                            extra={"path": img_path})
            
            return thresh
        except Exception as e:
            self.logger.exception(f"Error preprocessing image: {str(e)}", 
                                 extra={"path": img_path})
            return None
    
    def extract_text(self, img_path: str) -> Dict[str, Any]:
        """
        Extract text from an image.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Dictionary with extracted text elements and page information
        """
        start_time = time.time()
        processed = self.preprocess(img_path)
        if processed is None:
            self.logger.warning(f"Returning empty OCR result due to preprocessing failure: {img_path}")
            return {'text_elements': [], 'page': {'width': 0, 'height': 0}}

        try:
            # Extract text using Tesseract
            data = pytesseract.image_to_data(
                processed,
                output_type=Output.DICT,
                config=self.tesseract_config,
                timeout=30
            )
            
            # Get image dimensions
            with Image.open(img_path) as img:
                width, height = img.size
            
            # Filter and normalize text elements
            elements = []
            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                text = data['text'][i].strip()
                
                if conf > self.confidence_threshold and text:
                    elements.append({
                        'text': text,
                        'bbox': [
                            data['left'][i] / width,
                            data['top'][i] / height,
                            (data['left'][i] + data['width'][i]) / width,
                            (data['top'][i] + data['height'][i]) / height
                        ],
                        'conf': float(conf) / 100
                    })
            
            # Log extraction stats
            self.logger.info(f"OCR extraction completed in {time.time() - start_time:.2f}s", 
                            extra={"path": img_path, "elements_count": len(elements)})
            
            return {
                'text_elements': elements,
                'page': {'width': width, 'height': height}
            }
        except Exception as e:
            self.logger.exception(f"Error in OCR extraction: {str(e)}", 
                                 extra={"path": img_path})
            return {'text_elements': [], 'page': {'width': 0, 'height': 0}}


# Model Management Module
# ------------------------------

import torch
import torch.nn as nn
import pickle
import json
from transformers import AutoTokenizer, AutoModel

class DocClassifierModel(nn.Module):
    """Document classifier model combining BERT with extracted features."""
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased", feature_size: int = 5):
        """
        Initialize the document classifier.
        
        Args:
            model_name: Name of the pretrained language model
            feature_size: Size of the feature vector
        """
        super().__init__()
        self.model_name = model_name
        self.feature_size = feature_size
        
        # Load pretrained language model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification layer (output size can be configured)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 64, 2)
        
        self.logger = logging.getLogger("DocClassifierModel")
        self.logger.info(f"Model initialized with {model_name}, feature size: {feature_size}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            features: Extracted document features
            
        Returns:
            Classification logits
        """
        # Process text with BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Process features
        processed_features = self.feature_processor(features.float())
        
        # Combine text and feature representations
        combined = torch.cat([cls_output, processed_features], dim=1)
        
        # Classification
        return self.classifier(combined)
    
    def predict(self, feature_vector):
        """
        Predict document type from feature vector.
        
        Args:
            feature_vector: Vector of extracted features
            
        Returns:
            Prediction probabilities
        """
        # This is a simplified prediction method
        # In practice, you'd need to handle text processing too
        device = next(self.parameters()).device
        features = torch.tensor(feature_vector, dtype=torch.float32).to(device)
        
        # Create dummy text inputs (should be replaced with real text in practice)
        input_ids = torch.zeros((1, 10), dtype=torch.long).to(device)
        attention_mask = torch.zeros((1, 10), dtype=torch.long).to(device)
        
        # Get predictions
        with torch.no_grad():
            logits = self(input_ids, attention_mask, features)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
    
    def set_num_classes(self, num_classes: int):
        """
        Update the classifier for the number of document classes.
        
        Args:
            num_classes: Number of document classes
        """
        # Get the input size from the current classifier
        in_features = self.classifier.in_features
        
        # Replace the classifier layer
        self.classifier = nn.Linear(in_features, num_classes)
        self.logger.info(f"Updated classifier output size to {num_classes}")
    
    def save(self, path: str, save_weights_only: bool = False):
        """
        Save model to file.
        
        Args:
            path: Path to save the model
            save_weights_only: Whether to save only the weights (not the architecture)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if save_weights_only:
            torch.save(self.state_dict(), path)
        else:
            metadata = {
                'model_name': self.model_name,
                'feature_size': self.feature_size,
                'datetime': datetime.now().isoformat()
            }
            torch.save({
                'metadata': metadata,
                'state_dict': self.state_dict()
            }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> 'DocClassifierModel':
        """
        Load model from file.
        
        Args:
            path: Path to the saved model
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        logger = logging.getLogger("DocClassifierModel")
        
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different save formats
        if isinstance(checkpoint, dict) and 'metadata' in checkpoint:
            # Full save with metadata
            metadata = checkpoint['metadata']
            state_dict = checkpoint['state_dict']
            model = cls(
                model_name=metadata['model_name'],
                feature_size=metadata['feature_size']
            )
        else:
            # Weights-only save
            state_dict = checkpoint
            model = cls()
        
        model.load_state_dict(state_dict)
        model.to(device)
        
        logger.info(f"Model loaded from {path}")
        return model


class ModelLoader:
    """Manages loading and updating document classification models."""
    
    def __init__(self, config_manager):
        """
        Initialize model loader.
        
        Args:
            config_manager: Configuration manager
        """
        self.config = config_manager
        self.model_cache = {}
        self.tokenizer_cache = {}
        self.logger = logging.getLogger("ModelLoader")
    
    def load_active_model(self):
        """
        Load the currently active model.
        
        Returns:
            Loaded model, or None if no active model
        """
        model_config = self.config.get_active_model_config()
        if not model_config:
            self.logger.warning("No active model configuration found")
            return None
        
        # Check if model is already loaded
        model_id = model_config["id"]
        if model_id in self.model_cache:
            self.logger.info(f"Using cached model {model_id}")
            return self.model_cache[model_id]
        
        # Get model parameters
        model_type = model_config.get("model_type", "bert")
        model_path = model_config.get("model_path")
        model_name = model_config.get("pretrained_model_name", "bert-base-multilingual-cased")
        hyperparams = model_config.get("hyperparameters", {})
        
        # Load the model
        if model_type == "bert" and model_path:
            try:
                # Try to load from saved model path
                device = hyperparams.get("device", "cpu")
                model = DocClassifierModel.load(model_path, device)
                
                # Update number of classes
                num_classes = len(self.config.get_document_types())
                if model.classifier.out_features != num_classes:
                    model.set_num_classes(num_classes)
                
                self.model_cache[model_id] = model
                self.logger.info(f"Loaded model from {model_path}")
                return model
            except Exception as e:
                self.logger.error(f"Error loading model from {model_path}: {str(e)}")
                return None
        else:
            self.logger.warning(f"Unsupported model type: {model_type}")
            return None
    
    def load_tokenizer(self):
        """
        Load the tokenizer for the active model.
        
        Returns:
            Tokenizer for the active model
        """
        model_config = self.config.get_active_model_config()
        if not model_config:
            self.logger.warning("No active model configuration found")
            return None
        
        model_id = model_config["id"]
        if model_id in self.tokenizer_cache:
            return self.tokenizer_cache[model_id]
        
        model_name = model_config.get("pretrained_model_name", "bert-base-multilingual-cased")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer_cache[model_id] = tokenizer
            self.logger.info(f"Loaded tokenizer for {model_name}")
            return tokenizer
        except Exception as e:
            self.logger.error(f"Error loading tokenizer for {model_name}: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear model and tokenizer cache."""
        self.model_cache.clear()
        self.tokenizer_cache.clear()
        self.logger.info("Cleared model and tokenizer cache")


# API Module
# ------------------------------

from flask import Flask, request, jsonify, send_from_directory
import tempfile
from werkzeug.utils import secure_filename

class DocumentClassifierAPI:
    """API for document classification."""
    
    def __init__(self, config_manager, ocr, classifier, admin_api):
        """
        Initialize the API.
        
        Args:
            config_manager: Configuration manager
            ocr: OCR processor
            classifier: Document classifier
            admin_api: Admin API for configuration management
        """
        self.config = config_manager
        self.ocr = ocr
        self.classifier = classifier
        self.admin = admin_api
        self.logger = logging.getLogger("ClassifierAPI")
        
        # Create Flask app
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up API routes."""
        # Document classification endpoint
        @self.app.route('/api/classify', methods=['POST'])
        def classify_document():
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "Empty filename"}), 400
            
            # Save file temporarily
            filename = secure_filename(file.filename)
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            
            try:
                # Process document
                ocr_data = self.ocr.extract_text(file_path)
                classification = self.classifier.classify(ocr_data)
                
                # Add document info
                classification["document_info"] = {
                    "filename": filename,
                    "elements_count": len(ocr_data.get("text_elements", [])),
                    "process_id": str(uuid.uuid4())
                }
                
                return jsonify(classification)
            except Exception as e:
                self.logger.exception(f"Error classifying document: {str(e)}")
                return jsonify({"error": str(e)}), 500
            finally:
                # Clean up
                try:
                    os.remove(file_path)
                    os.rmdir(temp_dir)
                except:
                    pass
        
        # Document type management endpoints
        @self.app.route('/api/admin/document-types', methods=['GET'])
        def get_document_types():
            return jsonify(self.config.get_document_types())
        
        @self.app.route('/api/admin/document-types', methods=['POST'])
        def add_document_type():
            data = request.json
            if not data or not data.get('name'):
                return jsonify({"error": "Name is required"}), 400
            
            result = self.admin.add_document_type(
                data['name'], 
                data.get('description', '')
            )
            return jsonify(result)
        
        @self.app.route('/api/admin/document-types/<doc_type_id>', methods=['PUT'])
        def update_document_type(doc_type_id):
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            result = self.admin.update_document_type(
                doc_type_id, 
                data.get('name'), 
                data.get('description'), 
                data.get('active')
            )
            
            if not result:
                return jsonify({"error": "Document type not found"}), 404
            
            return jsonify(result)
        
        # Keyword management endpoints
        @self.app.route('/api/admin/keywords', methods=['GET'])
        def get_keywords():
            doc_type_id = request.args.get('document_type_id')
            language = request.args.get('language')
            
            if doc_type_id:
                return jsonify(self.config.get_keywords_for_document(doc_type_id, language))
            else:
                # Get all keywords
                keywords = []
                for doc_type in self.config.get_document_types():
                    doc_keywords = self.config.get_keywords_for_document(doc_type['id'])
                    for kw in doc_keywords:
                        kw['document_type_id'] = doc_type['id']
                        kw['document_type_name'] = doc_type['name']
                        keywords.append(kw)
                return jsonify(keywords)
        
        @self.app.route('/api/admin/keywords', methods=['POST'])
        def add_keyword():
            data = request.json
            if not data or not data.get('document_type_id') or not data.get('term'):
                return jsonify({"error": "document_type_id and term are required"}), 400
            
            result = self.admin.add_keyword(
                data['document_type_id'],
                data['term'],
                data.get('language', 'fr'),
                data.get('weight', 1.0)
            )
            return jsonify(result)
        
        # Feature management endpoints
        @self.app.route('/api/admin/features', methods=['GET'])
        def get_features():
            return jsonify(self.config.get_all_features())
        
        @self.app.route('/api/admin/features', methods=['POST'])
        def add_feature():
            data = request.json
            if not data or not data.get('name') or not data.get('extractor_type'):
                return jsonify({"error": "name and extractor_type are required"}), 400
            
            result = self.admin.add_feature(
                data['name'],
                data.get('description', ''),
                data['extractor_type'],
                data.get('extraction_parameters', {})
            )
            return jsonify(result)
        
        # Rule management endpoints
        @self.app.route('/api/admin/rules', methods=['GET'])
        def get_rules():
            doc_type_id = request.args.get('document_type_id')
            
            if doc_type_id:
                return jsonify(self.config.get_rules_for_document(doc_type_id))
            else:
                return jsonify(self.config.get_all_rules())
        
        @self.app.route('/api/admin/rules', methods=['POST'])
        def add_rule():
            data = request.json
            if not data or not data.get('document_type_id') or not data.get('feature_id'):
                return jsonify({"error": "document_type_id and feature_id are required"}), 400
            
            result = self.admin.add_rule(
                data['document_type_id'],
                data['feature_id'],
                data.get('operator', '>'),
                data.get('threshold', 0.0),
                data.get('weight', 1.0)
            )
            return jsonify(result)
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server."""
        self.logger.info(f"Starting API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

class AdminAPI:
    """API for managing document classification configuration."""
    
    def __init__(self, db_connection, config_manager):
        """
        Initialize admin API.
        
        Args:
            db_connection: Database connection
            config_manager: Configuration manager
        """
        self.db = db_connection
        self.config = config_manager
        self.logger = logging.getLogger("AdminAPI")
    
    def add_document_type(self, name, description):
        """
        Add a new document type.
        
        Args:
            name: Document type name
            description: Document type description
            
        Returns:
            Created document type
        """
        doc_type = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "active": True
        }
        
        self.db.insert("DocumentTypes", doc_type)
        self.config.refresh()  # Refresh configuration
        
        self.logger.info(f"Added document type: {name}")
        return doc_type
    
    def update_document_type(self, doc_type_id, name=None, description=None, active=None):
        """
        Update a document type.
        
        Args:
            doc_type_id: Document type ID
            name: New name (optional)
            description: New description (optional)
            active: New active status (optional)
            
        Returns:
            Updated document type, or None if not found
        """
        # Get current document type
        doc_type = self.config.get_document_type(doc_type_id)
        if not doc_type:
            self.logger.warning(f"Document type not found: {doc_type_id}")
            return None
        
        # Update fields
        update = {"updated_at": datetime.now()}
        if name is not None:
            update["name"] = name
        if description is not None:
            update["description"] = description
        if active is not None:
            update["active"] = active
        
        # Update in database
        self.db.update("DocumentTypes", {"id": doc_type_id}, update)
        self.config.refresh()  # Refresh configuration
        
        # Get updated document type
        updated_doc_type = self.config.get_document_type(doc_type_id)
        self.logger.info(f"Updated document type: {doc_type_id}")
        
        return updated_doc_type
    
    def add_keyword(self, doc_type_id, term, language="fr", weight=1.0):
        """
        Add a keyword for a document type.
        
        Args:
            doc_type_id: Document type ID
            term: Keyword term
            language: Language code (default: fr)
            weight: Keyword weight (default: 1.0)
            
        Returns:
            Created keyword
        """
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
        self.config.refresh()  # Refresh configuration
        
        self.logger.info(f"Added keyword '{term}' for document type {doc_type_id}")
        return keyword
    
    def add_feature(self, name, description, extractor_type, extraction_parameters=None):
        """
        Add a feature definition.
        
        Args:
            name: Feature name
            description: Feature description
            extractor_type: Type of feature extractor
            extraction_parameters: Parameters for extractor (optional)
            
        Returns:
            Created feature definition
        """
        feature = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "extractor_type": extractor_type,
            "extraction_parameters": extraction_parameters or {},
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        self.db.insert("FeatureDefinitions", feature)
        self.config.refresh()  # Refresh configuration
        
        self.logger.info(f"Added feature definition: {name}")
        return feature
    
    def add_rule(self, document_type_id, feature_id, operator=">", threshold=0.0, weight=1.0):
        """
        Add a classification rule.
        
        Args:
            document_type_id: Document type ID
            feature_id: Feature ID
            operator: Comparison operator (default: >)
            threshold: Comparison threshold (default: 0.0)
            weight: Rule weight (default: 1.0)
            
        Returns:
            Created rule
        """
        rule = {
            "id": str(uuid.uuid4()),
            "document_type_id": document_type_id,
            "feature_id": feature_id,
            "operator": operator,
            "threshold": threshold,
            "weight": weight,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        self.db.insert("ClassificationRules", rule)
        self.config.refresh()  # Refresh configuration
        
        self.logger.info(f"Added classification rule for document type {document_type_id}")
        return rule
    
    def add_model_config(self, name, model_type, pretrained_model_name, hyperparameters=None, feature_importances=None):
        """
        Add a model configuration.
        
        Args:
            name: Model configuration name
            model_type: Type of model
            pretrained_model_name: Name of pretrained model
            hyperparameters: Model hyperparameters (optional)
            feature_importances: Feature importance weights (optional)
            
        Returns:
            Created model configuration
        """
        model_config = {
            "id": str(uuid.uuid4()),
            "name": name,
            "model_type": model_type,
            "pretrained_model_name": pretrained_model_name,
            "hyperparameters": hyperparameters or {},
            "feature_importances": feature_importances or {},
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "active": True
        }
        
        # Deactivate other model configurations
        self.db.update("ModelConfigurations", {"active": True}, {"active": False})
        
        # Insert new model configuration
        self.db.insert("ModelConfigurations", model_config)
        self.config.refresh()  # Refresh configuration
        
        self.logger.info(f"Added model configuration: {name}")
        return model_config