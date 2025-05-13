import requests
import json
import base64
import os
from datetime import datetime
import logging
import sys
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import time
import math

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
        
        # Create output directories structure
        self.create_directory_structure()
        
    def create_directory_structure(self):
        """Create the complete directory structure for the pipeline"""
        directories = [
            os.path.join(self.output_dir, 'generated_data', 'bill_payments_a4'),
            os.path.join(self.output_dir, 'generated_data', 'bill_payments_80mm'),
            os.path.join(self.output_dir, 'generated_data', 'commission_statements'),
            
            os.path.join(self.output_dir, 'image_data', 'bill_payments_a4'),
            os.path.join(self.output_dir, 'image_data', 'bill_payments_80mm'),
            os.path.join(self.output_dir, 'image_data', 'commission_statements'),
            
            os.path.join(self.output_dir, 'augmented_data', 'bill_payments_a4'),
            os.path.join(self.output_dir, 'augmented_data', 'bill_payments_80mm'),
            os.path.join(self.output_dir, 'augmented_data', 'commission_statements'),
            
            os.path.join(self.output_dir, 'processed_data', 'bill_payments_a4'),
            os.path.join(self.output_dir, 'processed_data', 'bill_payments_80mm'),
            os.path.join(self.output_dir, 'processed_data', 'commission_statements'),
            
            os.path.join(self.output_dir, 'raw_data', 'bill_payments_a4'),
            os.path.join(self.output_dir, 'raw_data', 'bill_payments_80mm'),
            os.path.join(self.output_dir, 'raw_data', 'commission_statements'),
            
            os.path.join(self.output_dir, 'training_data', 'train', 'bill_payments_a4'),
            os.path.join(self.output_dir, 'training_data', 'train', 'bill_payments_80mm'),
            os.path.join(self.output_dir, 'training_data', 'train', 'commission_statements'),
            
            os.path.join(self.output_dir, 'training_data', 'validation', 'bill_payments_a4'),
            os.path.join(self.output_dir, 'training_data', 'validation', 'bill_payments_80mm'),
            os.path.join(self.output_dir, 'training_data', 'validation', 'commission_statements'),
            
            os.path.join(self.output_dir, 'training_data', 'test', 'bill_payments_a4'),
            os.path.join(self.output_dir, 'training_data', 'test', 'bill_payments_80mm'),
            os.path.join(self.output_dir, 'training_data', 'test', 'commission_statements'),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
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
            page_size = self.config.get('page_size', 100)
            max_pdfs = self.config.get('max_pdfs', 2 * page_size)
            max_pages = math.ceil(max_pdfs / page_size)

            while page < total_pages and page < max_pages:
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
                
                # Calculate total_pages on first response
                if page == 0 and 'lngArg1' in data:
                    total_objects = data['lngArg1']
                    total_pages = math.ceil(total_objects / page_size)

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
                pdf_path = os.path.join(self.output_dir, 'generated_data', folder, pdf_filename)
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
            page_size = self.config.get('page_size', 100)
            max_pdfs = self.config.get('max_pdfs', 2 * page_size)
            max_pages = math.ceil(max_pdfs / page_size)

            while page < total_pages and page < max_pages:
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
                
                # Calculate total_pages on first response
                if page == 0 and 'lngArg1' in data:
                    total_items = data['lngArg1']
                    total_pages = math.ceil(total_items / page_size)

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
            pdf_path = os.path.join(self.output_dir, 'generated_data', 'commission_statements', pdf_filename)
            
            with open(pdf_path, 'wb') as f:
                f.write(base64.b64decode(pdf_data))
            
            logger.info(f"Successfully generated invoice for statement {statement_id}: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to generate invoice for statement {statement_id}: {str(e)}")
            return None

    def convert_pdf_to_images(self):
        """1: Convert all PDFs to images"""
        logger.info("------------------------------------------------------------------------------------")
        logger.info("STEP 1: Converting PDFs to images")
        logger.info("------------------------------------------------------------------------------------")
        
        pdf_folders = ['bill_payments_a4', 'bill_payments_80mm', 'commission_statements']
        
        for folder in pdf_folders:
            pdf_dir = os.path.join(self.output_dir, 'generated_data', folder)
            image_dir = os.path.join(self.output_dir, 'image_data', folder)
            
            pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
            logger.info(f"Processing {len(pdf_files)} PDFs from {folder}")
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                
                try:
                    # Convert PDF to images
                    images = convert_from_path(pdf_path, dpi=300)
                    
                    # Save each page as an image
                    for i, image in enumerate(images):
                        image_filename = pdf_file.replace('.pdf', f'_page_{i+1}.png')
                        image_path = os.path.join(image_dir, image_filename)
                        image.save(image_path, 'PNG')
                        logger.info(f"Saved image: {image_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to convert {pdf_file}: {str(e)}")

    def augment_images(self):
        """2: Apply data augmentation techniques to images"""
        logger.info("------------------------------------------------------------------------------------")
        logger.info("STEP 2: Augmenting images")
        logger.info("------------------------------------------------------------------------------------")
        
        image_folders = ['bill_payments_a4', 'bill_payments_80mm', 'commission_statements']
        
        for folder in image_folders:
            image_dir = os.path.join(self.output_dir, 'image_data', folder)
            augmented_dir = os.path.join(self.output_dir, 'augmented_data', folder)
            
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
            logger.info(f"Augmenting {len(image_files)} images from {folder}")
            
            for image_file in image_files:
                image_path = os.path.join(image_dir, image_file)
                
                try:
                    # Open image
                    img = Image.open(image_path)
                    
                    # Create multiple augmented versions
                    augmentations = [
                        ('original', img),
                        ('rotated', self.rotate_image(img)),
                        ('noisy', self.add_noise(img)),
                        ('blur', self.add_blur(img)),
                        ('brightness', self.adjust_brightness(img)),
                        ('contrast', self.adjust_contrast(img)),
                        ('perspective', self.perspective_transform(img)),
                        ('print_artifact', self.add_print_artifacts(img))
                    ]
                    
                    # Save all augmented versions
                    for aug_type, aug_img in augmentations:
                        aug_filename = image_file.replace('.png', f'_{aug_type}.png')
                        aug_path = os.path.join(augmented_dir, aug_filename)
                        aug_img.save(aug_path, 'PNG')
                        logger.info(f"Saved augmented image: {aug_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to augment {image_file}: {str(e)}")

    def rotate_image(self, img):
        """Rotate image slightly (±5°)"""
        angle = random.uniform(-5, 5)
        return img.rotate(angle, fillcolor='white')

    def add_noise(self, img):
        """Add noise to simulate poor scan quality"""
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
        noisy_img = cv2.add(img_array, noise)
        return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))

    def add_blur(self, img):
        """Add blur to simulate poor quality scan"""
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2)))

    def adjust_brightness(self, img):
        """Adjust brightness randomly"""
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.8, 1.2)
        return enhancer.enhance(factor)

    def adjust_contrast(self, img):
        """Adjust contrast randomly"""
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.8, 1.2)
        return enhancer.enhance(factor)

    def perspective_transform(self, img):
        """Apply perspective transformation to simulate skewed scanning"""
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Define source points
        src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        
        # Apply random perspective distortion
        distortion = 0.05
        dst_points = src_points + np.random.uniform(-width*distortion, width*distortion, src_points.shape).astype(np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        transformed = cv2.warpPerspective(img_array, matrix, (width, height), borderValue=(255, 255, 255))
        
        return Image.fromarray(transformed)

    def add_print_artifacts(self, img):
        """Add print artifacts like stains or folds"""
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Add random stains
        for _ in range(random.randint(0, 3)):
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            radius = random.randint(10, 50)
            color = random.randint(200, 240)  # Light gray stains
            cv2.circle(img_array, (center_x, center_y), radius, (color, color, color), -1)
        
        # Simulate fold lines
        if random.random() > 0.5:
            fold_x = random.randint(int(width*0.3), int(width*0.7))
            cv2.line(img_array, (fold_x, 0), (fold_x, height), (180, 180, 180), 2)
        
        return Image.fromarray(img_array)

    def preprocess_images(self):
        """3: Preprocess augmented images for training"""
        logger.info("------------------------------------------------------------------------------------")
        logger.info("STEP 3: Preprocessing images")
        logger.info("------------------------------------------------------------------------------------")
        
        augmented_folders = ['bill_payments_a4', 'bill_payments_80mm', 'commission_statements']
        
        for folder in augmented_folders:
            augmented_dir = os.path.join(self.output_dir, 'augmented_data', folder)
            processed_dir = os.path.join(self.output_dir, 'processed_data', folder)
            
            image_files = [f for f in os.listdir(augmented_dir) if f.endswith('.png')]
            logger.info(f"Processing {len(image_files)} images from {folder}")
            
            for image_file in image_files:
                image_path = os.path.join(augmented_dir, image_file)
                
                try:
                    # Open image
                    img = cv2.imread(image_path)
                    
                    # Resize to standard size
                    # target_size = standard_sizes[folder]
                    # resized = cv2.resize(img, target_size)

                    # Convert to grayscale (optional)
                    if self.config.get('convert_to_grayscale', True):
                        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        processed = img
                    
                    # Normalize pixel values
                    normalized = processed.astype(np.float32) / 255.0
                    
                    # Save processed image
                    # processed_filename = image_file.replace('.png', '_processed.npy')
                    # processed_path = os.path.join(processed_dir, processed_filename)
                    # np.save(processed_path, normalized)
                    
                    # Also save as image for visualization
                    vis_filename = image_file.replace('.png', '_processed.png')
                    vis_path = os.path.join(processed_dir, vis_filename)
                    cv2.imwrite(vis_path, (normalized * 255).astype(np.uint8))
                    
                    logger.info(f"Processed and saved: {vis_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {image_file}: {str(e)}")

    def copy_raw_data(self):
        """4: Copy original PDFs to raw data folder"""
        logger.info("------------------------------------------------------------------------------------")
        logger.info("STEP 4: Copying raw data")
        logger.info("------------------------------------------------------------------------------------")
        
        pdf_folders = ['bill_payments_a4', 'bill_payments_80mm', 'commission_statements']
        
        for folder in pdf_folders:
            source_dir = os.path.join(self.output_dir, 'generated_data', folder)
            raw_dir = os.path.join(self.output_dir, 'raw_data', folder)
            
            pdf_files = [f for f in os.listdir(source_dir) if f.endswith('.pdf')]
            logger.info(f"Copying {len(pdf_files)} PDFs from {folder} to raw data")
            
            for pdf_file in pdf_files:
                source_path = os.path.join(source_dir, pdf_file)
                dest_path = os.path.join(raw_dir, pdf_file)
                
                try:
                    shutil.copy2(source_path, dest_path)
                    logger.info(f"Copied: {pdf_file}")
                except Exception as e:
                    logger.error(f"Failed to copy {pdf_file}: {str(e)}")

    def prepare_training_data(self):
        """5: Prepare training, validation, and test datasets"""
        logger.info("------------------------------------------------------------------------------------")
        logger.info("STEP 5: Preparing training datasets")
        logger.info("------------------------------------------------------------------------------------")
        
        processed_folders = ['bill_payments_a4', 'bill_payments_80mm', 'commission_statements']
        
        all_files = []
        all_labels = []
        
        # Collect all processed files
        for i, folder in enumerate(processed_folders):
            processed_dir = os.path.join(self.output_dir, 'processed_data', folder)
            files = [f for f in os.listdir(processed_dir) if f.endswith('.png')]
            
            for file in files:
                all_files.append(os.path.join(processed_dir, file))
                all_labels.append(i)
        
        logger.info(f"Total files: {len(all_files)}")
        
        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_files, all_labels, test_size=0.15, random_state=42, stratify=all_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
        )  # 0.176 * 0.85 ≈ 0.15, so we get 70-15-15 split
        
        # Copy files to respective directories
        datasets = [
            (X_train, y_train, 'train'),
            (X_val, y_val, 'validation'),
            (X_test, y_test, 'test')
        ]
        
        for X, y, split_name in datasets:
            logger.info(f"Preparing {split_name} set with {len(X)} samples")
            
            for file_path, label in zip(X, y):
                # Determine the category
                category = processed_folders[label]
                
                # Define destination path
                filename = os.path.basename(file_path)
                dest_dir = os.path.join(self.output_dir, 'training_data', split_name, category)
                dest_path = os.path.join(dest_dir, filename)
                
                # Copy file
                try:
                    shutil.copy2(file_path, dest_path)
                    
                    # Also copy the corresponding PNG for visualization
                    png_file = file_path.replace('.npy', '.png')
                    if os.path.exists(png_file):
                        png_dest = dest_path.replace('.npy', '.png')
                        shutil.copy2(png_file, png_dest)
                        
                except Exception as e:
                    logger.error(f"Failed to copy {filename}: {str(e)}")
        
        # Create metadata file
        metadata = {
            'total_samples': len(all_files),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'categories': processed_folders,
            'split_ratios': '70-15-15'
        }
        
        metadata_path = os.path.join(self.output_dir, 'training_data', 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Created metadata file: {metadata_path}")

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='INWI Report Generator - Process PDFs through complete ML pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # API Configuration
    parser.add_argument('--login', type=str, default='Alia.Gueddari', help='Admin user login')
    parser.add_argument('--password', type=str, default='Azerty11@!', help='Admin user password')
    parser.add_argument('--auth-url', type=str, default='http://localhost:8080/inwi_lean_agent', help='Authentication server URL')
    parser.add_argument('--report-url', type=str, default='http://localhost:8080/inwi_lean_report', help='Report server URL')
    parser.add_argument('--agent-url', type=str, default='http://localhost:8080/inwi_lean_agent', help='Agent server URL')    
    # Processing Options
    parser.add_argument('--output-dir', type=str, default='', help='Output directory for all generated files')
    parser.add_argument('--page-size', type=int, default=100, help='Page size for API pagination')
    parser.add_argument('--max-pdfs', type=int, default=100, help='Maximum number of PDFs to generate (default: 2 * page_size)')
    parser.add_argument('--grayscale', action='store_true', default=True, help='Convert images to grayscale during preprocessing')
    # Pipeline Control
    parser.add_argument('--skip-pdf-generation', action='store_true', help='Skip PDF generation and use existing PDFs')
    parser.add_argument('--skip-image-conversion', action='store_true', help='Skip PDF to image conversion')
    parser.add_argument('--skip-augmentation', action='store_true', help='Skip data augmentation')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip image preprocessing')
    parser.add_argument('--skip-raw-backup', action='store_false', help='Skip raw data backup')
    parser.add_argument('--skip-training-split', action='store_true', help='Skip training data preparation')      
    # Advanced Options
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF to image conversion')
    parser.add_argument('--augmentation-factor', type=int, default=8, help='Number of augmented versions per image')
    parser.add_argument('--train-split', type=float, default=0.7, help='Training data percentage')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation data percentage')
    parser.add_argument('--test-split', type=float, default=0.15, help='Test data percentage')    
    # Logging and Debug
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually doing it')   
    args = parser.parse_args()    
    # Validate split ratios
    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 0.001:
        logger.error(f"Split ratios must sum to 1.0 (current: {total_split})")
        sys.exit(1)

    # Configuration settings from arguments
    config = {
        'admin_user_login': args.login,
        'admin_user_password': args.password,
        'app_frontend_code': 'inwi_wakil_web',
        'app_frontend_institution_code': 'INWI',
        'app_frontend_version': '2.10.0',
        'auth_server_url': args.auth_url,
        'report_server_url': args.report_url,
        'agent_server_url': args.agent_url,
        'contract_id': 1000,
        'session_role_code': 'BOC-Dir',
        'page_size': args.page_size,
        'max_pdfs': args.max_pdfs if args.max_pdfs is not None else 2 * args.page_size,
        'output_directory': args.output_dir,
        'convert_to_grayscale': args.grayscale,
        'dpi': args.dpi,
        'augmentation_factor': args.augmentation_factor,
        'train_split': args.train_split,
        'val_split': args.val_split,
        'test_split': args.test_split,
        'verbose': args.verbose,
        'dry_run': args.dry_run,
        'skip_pdf_generation': args.skip_pdf_generation,
        'skip_image_conversion': args.skip_image_conversion,
        'skip_augmentation': args.skip_augmentation,
        'skip_preprocessing': args.skip_preprocessing,
        'skip_raw_backup': args.skip_raw_backup,
        'skip_training_split': args.skip_training_split,
    }

   
    # Set logging level based on verbosity
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)
    
    # Print configuration if verbose
    if args.verbose:
        logger.info("Configuration:")
        for key, value in config.items():
            if key != 'admin_user_password':
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {'*' * len(value)}")
    
    # Start timer
    start_time = time.time()
    
    try:
        # Create generator instance
        generator = DataGenerator(config)
        
        logger.info("------------------------------------------------------------------------------------")
        logger.info("DATA-GENERATOR: Starting process")
        logger.info("------------------------------------------------------------------------------------")
        
        success = True
        
        if not args.skip_pdf_generation:
            logger.info("Stage 1: PDF Generation")
            try:
                generator.login()
                
                # Process bill payments
                bill_payments = generator.get_bill_payments()
                logger.info(f"Processing {len(bill_payments)} bill payments...")
                
                if not args.dry_run:
                    for payment in bill_payments:
                        payment_id = payment.get('id')
                        if payment_id:
                            generator.generate_bill_payment_receipt(payment_id)
                
                # Process commission statements
                commission_statements = generator.get_commission_statements()
                logger.info(f"Processing {len(commission_statements)} commission statements...")
                
                if not args.dry_run:
                    for statement in commission_statements:
                        statement_id = statement.get('id')
                        if statement_id:
                            generator.generate_commission_statement_invoice(statement_id)
                
                generator.logout()
                
            except Exception as e:
                logger.error(f"PDF generation failed: {str(e)}")
                success = False
        else:
            logger.info("Skipping PDF generation (using existing PDFs)")
        
        # Continue with other stages
        if success and not args.skip_image_conversion:
            logger.info("Stage 2: PDF to Image Conversion")
            try:
                if not args.dry_run:
                    generator.convert_pdf_to_images()
                else:
                    logger.info("Dry run: Would convert PDFs to images")
            except Exception as e:
                logger.error(f"Image conversion failed: {str(e)}")
                success = False
        
        if success and not args.skip_augmentation:
            logger.info("Stage 3: Data Augmentation")
            try:
                if not args.dry_run:
                    generator.augment_images()
                else:
                    logger.info("Dry run: Would augment images")
            except Exception as e:
                logger.error(f"Data augmentation failed: {str(e)}")
                success = False
        
        if success and not args.skip_preprocessing:
            logger.info("Stage 4: Image Preprocessing")
            try:
                if not args.dry_run:
                    generator.preprocess_images()
                else:
                    logger.info("Dry run: Would preprocess images")
            except Exception as e:
                logger.error(f"Preprocessing failed: {str(e)}")
                success = False
        
        if success and not args.skip_raw_backup:
            logger.info("Stage 5: Raw Data Backup")
            try:
                if not args.dry_run:
                    generator.copy_raw_data()
                else:
                    logger.info("Dry run: Would backup raw data")
            except Exception as e:
                logger.error(f"Raw data backup failed: {str(e)}")
                success = False
        
        if success and not args.skip_training_split:
            logger.info("Stage 6: Training Data Preparation")
            try:
                if not args.dry_run:
                    generator.prepare_training_data()
                else:
                    logger.info("Dry run: Would prepare training data")
            except Exception as e:
                logger.error(f"Training data preparation failed: {str(e)}")
                success = False
        
        # Calculate execution time
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("------------------------------------------------------------------------------------")
        if success:
            logger.info(f"DATA-GENERATOR: Process completed successfully in {duration:.2f} seconds")
        else:
            logger.info(f"DATA-GENERATOR: Process failed after {duration:.2f} seconds")
        logger.info("------------------------------------------------------------------------------------")
        
        # Print summary statistics
        if success and args.verbose:
            logger.info("\nSummary Statistics:")
            
            # Count files in each directory
            folders = [
                ('PDFs', ['generated_data/bill_payments_a4', 'generated_data/bill_payments_80mm', 'generated_data/commission_statements']),
                ('Images', ['image_data/bill_payments_a4', 'image_data/bill_payments_80mm', 'image_data/commission_statements']),
                ('Augmented', ['augmented_data/bill_payments_a4', 'augmented_data/bill_payments_80mm', 'augmented_data/commission_statements']),
                ('Processed', ['processed_data/bill_payments_a4', 'processed_data/bill_payments_80mm', 'processed_data/commission_statements']),
                ('Training', ['training_data/train', 'training_data/validation', 'training_data/test'])
            ]
            
            for stage_name, stage_folders in folders:
                total_files = 0
                for folder in stage_folders:
                    folder_path = os.path.join(args.output_dir, folder)
                    if os.path.exists(folder_path):
                        # Count files recursively
                        for root, dirs, files in os.walk(folder_path):
                            total_files += len(files)
                logger.info(f"  {stage_name}: {total_files} files")
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)
    