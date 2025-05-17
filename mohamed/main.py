# main.py - Main Application Module
# ------------------------------

import os
import sys
import logging
import logging.handlers
import argparse
from configparser import ConfigParser
from datetime import datetime

# Import components
from database_manager import get_db_connection
from config_manager import ConfigurationManager
from ocr_model_api import MoroccanOCR, ModelLoader, DocClassifierModel, AdminAPI, DocumentClassifierAPI
from feature_extraction_system import FeatureExtractorFactory, DynamicClassifier

def setup_logging(log_dir):
    """Set up logging configuration."""
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler (rotating log files)
    log_file = os.path.join(log_dir, f"doc_classifier_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5  # 10MB files, keep 5 backups
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def main():
    """Main application entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Moroccan Document Classifier")
    parser.add_argument("--config", default="config.ini", help="Path to configuration file")
    parser.add_argument("--log-dir", default="logs", help="Path to log directory")
    parser.add_argument("--port", type=int, default=5000, help="API server port")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--init-db", action="store_true", help="Initialize database with sample data")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting Moroccan Document Classifier")
    
    # Load configuration
    config_parser = ConfigParser()
    if os.path.exists(args.config):
        config_parser.read(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.warning(f"Configuration file {args.config} not found, using defaults")
    
    try:
        # Initialize database connection
        db_connection = get_db_connection()
        logger.info("Database connection initialized")
        
        # Initialize database if requested
        if args.init_db:
            from config_init import init_database
            init_database(db_connection, sample_data=True)
            logger.info("Database initialized with sample data")
        
        # Initialize configuration manager
        config_manager = ConfigurationManager(db_connection)
        logger.info("Configuration manager initialized")
        
        # Initialize OCR processor
        ocr = MoroccanOCR(config_manager)
        logger.info("OCR processor initialized")
        
        # Initialize feature extractor factory
        feature_factory = FeatureExtractorFactory(config_manager)
        logger.info("Feature extractor factory initialized")
        
        # Initialize model loader
        model_loader = ModelLoader(config_manager)
        logger.info("Model loader initialized")
        
        # Initialize classifier
        classifier = DynamicClassifier(config_manager, feature_factory, model_loader)
        logger.info("Document classifier initialized")
        
        # Initialize admin API
        admin_api = AdminAPI(db_connection, config_manager)
        logger.info("Admin API initialized")
        
        # Initialize API server
        api = DocumentClassifierAPI(config_manager, ocr, classifier, admin_api)
        logger.info("API server initialized")
        
        # Run API server
        logger.info(f"Starting API server on {args.host}:{args.port}")
        api.run(host=args.host, port=args.port, debug=args.debug)
        
    except Exception as e:
        logger.exception(f"Error starting application: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())