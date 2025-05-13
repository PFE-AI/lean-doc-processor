flowchart TD
    A[Document uploadé] --> B[Prétraitement d'image]
    B --> C[Classification CNN]
    C -->|Type de document| D[OCR]
    D -->|Texte extrait| E[Extraction LLM]
    D -->|Zones d'intérêt| E
    C -->|Type de document| E
    E -->|Données structurées| F[Validation]
    F -->|Règles métier| G[Validation syntaxique]
    F -->|Cohérence sémantique| H[Validation LLM]
    G --> I[Résultat final]
    H --> I
    
    subgraph "Traitement visuel (CNN)"
        B
        C
    end
    
    subgraph "Traitement textuel (LLM)"
        E
        H
    end