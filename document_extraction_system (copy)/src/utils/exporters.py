# src/utils/exporters.py
"""
Utility functions for exporting extracted data to various formats
"""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from src.base.models import ExtractedData, DocumentType


class DataExporter:
    """Export extracted data to various formats"""
    
    @staticmethod
    def to_json(extracted_data: ExtractedData, output_path: Path) -> None:
        """Export to JSON format"""
        data = {
            'document_type': extracted_data.document_type.value,
            'extracted_fields': extracted_data.extracted_fields,
            'validation_errors': extracted_data.validation_errors,
            'metadata': extracted_data.metadata,
            'processing_time': extracted_data.processing_time,
            'ocr_quality_score': extracted_data.ocr_quality_score
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def to_csv(extracted_data_list: List[ExtractedData], output_path: Path) -> None:
        """Export multiple extractions to CSV"""
        rows = []
        
        for data in extracted_data_list:
            row = {
                'document_type': data.document_type.value,
                'processing_time': data.processing_time,
                'ocr_quality_score': data.ocr_quality_score,
                'validation_errors_count': len(data.validation_errors),
                'is_valid': len(data.validation_errors) == 0
            }
            
            # Flatten extracted fields
            for key, value in data.extracted_fields.items():
                if isinstance(value, (str, int, float, bool)):
                    row[f'field_{key}'] = value
                elif isinstance(value, list):
                    row[f'field_{key}_count'] = len(value)
                elif isinstance(value, dict):
                    row[f'field_{key}_keys'] = ','.join(value.keys())
            
            rows.append(row)
        
        # Write to CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False, encoding='utf-8')
    
    @staticmethod
    def to_excel(extracted_data_list: List[ExtractedData], output_path: Path) -> None:
        """Export to Excel with multiple sheets"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for data in extracted_data_list:
                summary_data.append({
                    'Document Type': data.document_type.value,
                    'Processing Time': data.processing_time,
                    'OCR Quality': data.ocr_quality_score,
                    'Valid': len(data.validation_errors) == 0,
                    'Errors': ', '.join(data.validation_errors)
                })
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Separate sheets by document type
            by_type = {}
            for data in extracted_data_list:
                doc_type = data.document_type.value
                if doc_type not in by_type:
                    by_type[doc_type] = []
                by_type[doc_type].append(data.extracted_fields)
            
            for doc_type, fields_list in by_type.items():
                df = pd.DataFrame(fields_list)
                sheet_name = doc_type.replace('_', ' ').title()[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    @staticmethod
    def generate_report(extracted_data_list: List[ExtractedData], output_path: Path) -> None:
        """Generate a detailed HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Extraction Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .document { border: 1px solid #ccc; margin: 10px 0; padding: 15px; }
                .success { color: green; }
                .error { color: red; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Document Extraction Report</h1>
            <p>Generated on: {timestamp}</p>
            <p>Total documents processed: {total_docs}</p>
            
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Document Type</th>
                    <th>Count</th>
                    <th>Success Rate</th>
                    <th>Avg. Processing Time</th>
                    <th>Avg. OCR Quality</th>
                </tr>
                {summary_rows}
            </table>
            
            <h2>Document Details</h2>
            {document_details}
        </body>
        </html>
        """
        
        from datetime import datetime
        from collections import defaultdict
        
        # Group by document type
        by_type = defaultdict(list)
        for data in extracted_data_list:
            by_type[data.document_type.value].append(data)
        
        # Generate summary rows
        summary_rows = []
        for doc_type, docs in by_type.items():
            success_count = sum(1 for d in docs if len(d.validation_errors) == 0)
            success_rate = (success_count / len(docs) * 100) if docs else 0
            avg_time = sum(d.processing_time for d in docs) / len(docs) if docs else 0
            avg_quality = sum(d.ocr_quality_score or 0 for d in docs) / len(docs) if docs else 0
            
            summary_rows.append(f"""
                <tr>
                    <td>{doc_type}</td>
                    <td>{len(docs)}</td>
                    <td>{success_rate:.1f}%</td>
                    <td>{avg_time:.2f}s</td>
                    <td>{avg_quality:.2f}</td>
                </tr>
            """)
        
        # Generate document details
        document_details = []
        for i, data in enumerate(extracted_data_list):
            status = 'success' if len(data.validation_errors) == 0 else 'error'
            document_details.append(f"""
                <div class="document">
                    <h3>Document {i+1}: {data.document_type.value}</h3>
                    <p class="{status}">Status: {'Success' if status == 'success' else 'Failed'}</p>
                    <p>Processing time: {data.processing_time:.2f}s</p>
                    <p>OCR quality: {data.ocr_quality_score:.2f}</p>
                    
                    <h4>Extracted Fields:</h4>
                    <pre>{json.dumps(data.extracted_fields, indent=2, ensure_ascii=False)}</pre>
                    
                    {f'<h4>Validation Errors:</h4><ul>{"".join(f"<li>{e}</li>" for e in data.validation_errors)}</ul>' if data.validation_errors else ''}
                </div>
            """)
        
        # Fill template
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_docs=len(extracted_data_list),
            summary_rows=''.join(summary_rows),
            document_details=''.join(document_details)
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)