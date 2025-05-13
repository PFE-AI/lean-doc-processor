import requests
import json
import base64
import os
from datetime import datetime
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("data.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.token = None
        self.session = requests.Session()
        self.output_dir = config.get('output_directory', 'generated')
        
        # Create output directories if they don't exist
        os.makedirs(os.path.join(self.output_dir, 'bill_payments_a4'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'bill_payments_80mm'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'commission_statements'), exist_ok=True)
        
    def initialize_base_request(self, token=None):
        """Create a base request object similar to the Java implementation"""
        current_time = datetime.now().isoformat() + "Z"
        
        # Get the admin password
        if token is None:
            password = self.config.get('admin_user_password')
        
        # Create the base request object
        request = {
            "appInfo": {
                "appCode": self.config.get('app_frontend_code'),
                "appInstitutionCode": self.config.get('app_frontend_institution_code'),
                "appVersion": self.config.get('app_frontend_version')
            },
            "login": self.config.get('admin_user_login'),
            "messageInfo": {
                "transactionTime": current_time,
                "language": "Fr",
                "accessChannel": "WEB",
                "appRelease": "10"
            },
            "deviceInfo": {
                "deviceUuid": "default"
            }
        }
        
        # Add either password or token
        if token is None:
            request["password"] = password
        else:
            request["token"] = token
            
        # Add common fields used in requests
        request["page"] = 0
        request["pageSize"] = self.config.get('page_size', 100)
        request["sessionRoleCode"] = self.config.get('session_role_code', "BOC-Dir")
        request["contractId"] = self.config.get('contract_id', 1000)
        
        return request

    def login(self):
        """Login to the system and get a token"""
        logger.info("Attempting to login...")
        
        try:
            login_url = f"{self.config.get('auth_server_url')}/security/login"
            request = self.initialize_base_request()
            
            response = self.session.post(login_url, json=request)
            response_data = response.json()
            
            if (response_data is None or 
                response_data.get('response') is None or 
                response_data.get('response', {}).get('code') != "Accepted"):
                logger.error(f"Login failed. Response: {response_data}")
                raise Exception(f"Login failed for: {login_url}")
            
            self.token = response_data.get('person', {}).get('token')
            logger.info(f"Login successful. Received token: {self.token}")
            
            return self.token
            
        except Exception as e:
            logger.error(f"Login process failed: {str(e)}")
            raise

    def logout(self):
        """Logout from the system"""
        logger.info("Logging out...")
        
        try:
            logout_url = f"{self.config.get('auth_server_url')}/security/logout"
            request = self.initialize_base_request(self.token)
            
            response = self.session.post(logout_url, json=request)
            response_data = response.json()
            
            if (response_data is None or 
                response_data.get('response') is None or 
                response_data.get('response', {}).get('code') != "Accepted"):
                logger.error(f"Logout failed. Response: {response_data}")
                raise Exception(f"Logout failed for: {logout_url}")
            
            logger.info("Logout successful.")
            
        except Exception as e:
            logger.error(f"Logout process failed: {str(e)}")
            raise

    def get_bill_payments(self):
        """Get all bill payments with pagination"""
        logger.info("Fetching bill payments...")
        
        try:
            all_payments = []
            page = 0
            total_pages = 1  # Start with 1, will be updated after first request
            
            while page < total_pages:
                url = f"{self.config.get('report_server_url')}/gallery/institutionBillPaymentsGet"
                
                request = self.initialize_base_request(self.token)
                request["page"] = page
                request["strArg1"] = "F"
                request["blnArg1"] = True
                request["blnArg2"] = True
                request["blnArg3"] = True
                request["booleans"] = [False, False, False, False, False]
                
                response = self.session.post(url, json=request)
                data = response.json()
                
                if (data is None or 
                    data.get('response') is None or 
                    data.get('response', {}).get('code') != "Accepted"):
                    logger.error(f"Failed to get bill payments. Response: {data}")
                    raise Exception(f"Failed to get bill payments from: {url}")
                
                # Extract bill payments from the response
                payments = data.get('billPayments', [])
                all_payments.extend(payments)
                
                # Update pagination info
                total_pages = data.get('paging', {}).get('total', 1)
                logger.info(f"Fetched page {page+1} of {total_pages}, got {len(payments)} bill payments")
                
                page += 1
            
            logger.info(f"Successfully fetched {len(all_payments)} bill payments")
            return all_payments
            
        except Exception as e:
            logger.error(f"Failed to get bill payments: {str(e)}")
            raise

    def generate_bill_payment_receipt(self, payment_id):
        """Generate receipts in both A4 and 80mm formats for a specific bill payment"""
        formats = [("A4", "bill_payments_a4"), ("80mm", "bill_payments_80mm")]
        generated_paths = []

        for format_type, folder in formats:
            try:
                logger.info(f"Generating {format_type} receipt for bill payment ID: {payment_id}")

                url = f"{self.config.get('agent_server_url')}/gallery/billPaymentReceipt"
                request = self.initialize_base_request(self.token)
                request["id"] = payment_id
                request["blnArg1"] = True
                request["strArg1"] = format_type

                response = self.session.post(url, json=request)
                data = response.json()

                if (data is None or 
                    data.get('response') is None or 
                    data.get('response', {}).get('code') != "Accepted"):
                    logger.error(f"Failed to generate {format_type} receipt for payment {payment_id}. Response: {data}")
                    continue

                pdf_data = data.get('strArg1')
                if not pdf_data:
                    logger.warning(f"No PDF data returned for payment {payment_id} ({format_type})")
                    continue

                # Define PDF filename and path
                pdf_filename = f"bill_payment_receipt_{payment_id}_{format_type}.pdf"
                pdf_path = os.path.join(self.output_dir, folder, pdf_filename)
                os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

                with open(pdf_path, 'wb') as f:
                    f.write(base64.b64decode(pdf_data))

                logger.info(f"Successfully generated {format_type} receipt for payment {payment_id}: {pdf_path}")
                generated_paths.append(pdf_path)

            except Exception as e:
                logger.error(f"Exception generating {format_type} receipt for payment {payment_id}: {str(e)}")

        return generated_paths if generated_paths else None

    def get_commission_statements(self):
        """Get all commission statements with pagination"""
        logger.info("Fetching commission statements...")
        
        try:
            all_statements = []
            page = 0
            total_pages = 1  # Start with 1, will be updated after first request
            
            while page < total_pages:
                url = f"{self.config.get('agent_server_url')}/commission/commissionStatementsGet"
                
                request = self.initialize_base_request(self.token)
                request["page"] = page
                
                response = self.session.post(url, json=request)
                data = response.json()
                
                if (data is None or 
                    data.get('response') is None or 
                    data.get('response', {}).get('code') != "Accepted"):
                    logger.error(f"Failed to get commission statements. Response: {data}")
                    raise Exception(f"Failed to get commission statements from: {url}")
                
                # Extract commission statements from the response
                statements = data.get('commissionStatements', [])
                all_statements.extend(statements)
                
                # Update pagination info
                total_pages = data.get('paging', {}).get('total', 1)
                logger.info(f"Fetched page {page+1} of {total_pages}, got {len(statements)} commission statements")
                
                page += 1
            
            logger.info(f"Successfully fetched {len(all_statements)} commission statements")
            return all_statements
            
        except Exception as e:
            logger.error(f"Failed to get commission statements: {str(e)}")
            raise

    def generate_commission_statement_invoice(self, statement_id):
        """Generate an invoice for a specific commission statement"""
        logger.info(f"Generating invoice for commission statement ID: {statement_id}")
        
        try:
            url = f"{self.config.get('agent_server_url')}/commission/commissionStatementInvoicePrint"
            
            request = self.initialize_base_request(self.token)
            request["id"] = statement_id
            request["strArg1"] = self.config.get('session_role_code', "AE6000")
            
            response = self.session.post(url, json=request)
            data = response.json()
            
            if (data is None or 
                data.get('response') is None or 
                data.get('response', {}).get('code') != "Accepted"):
                logger.error(f"Failed to generate invoice for statement {statement_id}. Response: {data}")
                raise Exception(f"Failed to generate invoice from: {url}")
            
            # Extract the PDF content (Base64 encoded)
            pdf_data = data.get('strArg1')
            if not pdf_data:
                logger.warning(f"No PDF data returned for statement {statement_id}")
                return None
            
            # Save the PDF to disk
            pdf_filename = f"commission_statement_invoice_{statement_id}.pdf"
            pdf_path = os.path.join(self.output_dir, 'commission_statements', pdf_filename)
            
            with open(pdf_path, 'wb') as f:
                f.write(base64.b64decode(pdf_data))
            
            logger.info(f"Successfully generated invoice for statement {statement_id}: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to generate invoice for statement {statement_id}: {str(e)}")
            return None

    def process_all(self):
        """Process all bill payments and commission statements"""
        logger.info("------------------------------------------------------------------------------------")
        logger.info("INWI-REPORT-GENERATOR: Starting process")
        logger.info("------------------------------------------------------------------------------------")
        
        try:
            # Login to the system
            self.login()
            
            try:
                # Process bill payments
                bill_payments = self.get_bill_payments()
                logger.info(f"Processing {len(bill_payments)} bill payments...")
                
                for payment in bill_payments:
                    payment_id = payment.get('id')
                    if payment_id:
                        self.generate_bill_payment_receipt(payment_id)
                
                # Process commission statements
                commission_statements = self.get_commission_statements()
                logger.info(f"Processing {len(commission_statements)} commission statements...")
                
                for statement in commission_statements:
                    statement_id = statement.get('id')
                    if statement_id:
                        self.generate_commission_statement_invoice(statement_id)
                
                logger.info("Processing completed successfully")
                
            finally:
                # Always logout even if there's an error
                self.logout()
                
            logger.info("------------------------------------------------------------------------------------")
            logger.info("INWI-REPORT-GENERATOR: Process completed")
            logger.info("------------------------------------------------------------------------------------")
            
        except Exception as e:
            logger.error(f"An error occurred during processing: {str(e)}")
            return False
            
        return True

if __name__ == "__main__":
    
    # Configuration settings
    config = {
        'admin_user_login': 'Alia.Gueddari',
        'admin_user_password': 'Azerty11@!',
        'app_frontend_code': 'inwi_wakil_web',
        'app_frontend_institution_code': 'INWI',
        'app_frontend_version': '2.10.0',
        'auth_server_url': 'http://localhost:8080/inwi_lean_agent',
        'report_server_url': 'http://localhost:8080/inwi_lean_report',
        'agent_server_url': 'http://localhost:8080/inwi_lean_agent',
        'contract_id': 1000,
        'session_role_code': 'BOC-Dir',
        'page_size': 100,
        'output_directory': 'generated'
    }
    
    # Create and run the generator
    generator = DataGenerator(config)
    success = generator.process_all()
    
    # TODO : Image Data : Convert All PDFs to images
     Image Data
Tous les fichiers PDF générés sont convertis en images (format .png ou .jpeg) :

Une image par page

Résolution adaptée pour l’OCR ou un modèle de classification

Cette étape est essentielle pour préparer les données en vue d’un traitement par des modèles de vision par ordinateur ou d’OCR (reconnaissance optique de caractères).


    # TODO Augmentation Data 
Des techniques d’augmentation de données sont appliquées sur les images générées :

Rotation légère (±5°)

Zoom/recadrage

Ajout de bruit (simulant un scan flou)

Changement de contraste/luminosité

Ajout d’artefacts d’impression (effet de tache, pli)

L’objectif est de rendre les modèles plus robustes face à des documents variés (scannés, photographiés, de mauvaise qualité, etc.).

    # TODO Processed
Les images augmentées sont ensuite prétraitées pour l'entraînement :

Redimensionnement à une taille standard (par exemple 224x224 ou 512x512)

Normalisation des pixels

(optionnel) Conversion en niveaux de gris

Les fichiers sont ensuite labellisés en fonction de leur catégorie (reçu 80mm, reçu A4, relevé commission).


    # TODO Raw Data
Les documents originaux (PDF non traités) sont conservés séparément en tant que données brutes pour :

Références ultérieures

Vérification de cohérence

Génération de nouvelles variantes au besoin

    # TODO Training

Un modèle d’apprentissage automatique ou de deep learning est entraîné à partir des images prétraitées et labellisées. Selon le cas d’usage :

OCR pour extraire les données des documents

Classification pour identifier le type de document

Détection de zones (bounding boxes) pour segmenter les champs importants (montant, date, ID client, etc.)

Le dataset est divisé en :

Données d’entraînement

Données de validation

Données de test

    # Exit with appropriate code
    sys.exit(0 if success else -1)