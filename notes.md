# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Installer les packages essentiels
pip install tensorflow opencv-python pytesseract spacy transformers fastapi uvicorn