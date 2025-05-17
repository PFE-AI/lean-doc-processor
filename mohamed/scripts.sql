# Database-Driven Document Classifier - SQL Server Schema

-- Create DocumentTypes table
CREATE TABLE DocumentTypes (
    id NVARCHAR(36) PRIMARY KEY,
    name NVARCHAR(100) NOT NULL,
    description NVARCHAR(MAX),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    active BIT NOT NULL DEFAULT 1
);

-- Create Keywords table
CREATE TABLE Keywords (
    id NVARCHAR(36) PRIMARY KEY,
    document_type_id NVARCHAR(36) NOT NULL,
    term NVARCHAR(100) NOT NULL,
    language NVARCHAR(10) NOT NULL,
    weight FLOAT NOT NULL DEFAULT 1.0,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    FOREIGN KEY (document_type_id) REFERENCES DocumentTypes(id)
);

-- Create FeatureDefinitions table
CREATE TABLE FeatureDefinitions (
    id NVARCHAR(36) PRIMARY KEY,
    name NVARCHAR(100) NOT NULL,
    description NVARCHAR(MAX),
    extractor_type NVARCHAR(50) NOT NULL,
    extraction_parameters NVARCHAR(MAX),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

-- Create ClassificationRules table
CREATE TABLE ClassificationRules (
    id NVARCHAR(36) PRIMARY KEY,
    document_type_id NVARCHAR(36) NOT NULL,
    feature_id NVARCHAR(36) NOT NULL,
    operator NVARCHAR(10) NOT NULL,
    threshold FLOAT NOT NULL,
    weight FLOAT NOT NULL DEFAULT 1.0,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    FOREIGN KEY (document_type_id) REFERENCES DocumentTypes(id),
    FOREIGN KEY (feature_id) REFERENCES FeatureDefinitions(id)
);

-- Create ModelConfigurations table
CREATE TABLE ModelConfigurations (
    id NVARCHAR(36) PRIMARY KEY,
    name NVARCHAR(100) NOT NULL,
    model_type NVARCHAR(50) NOT NULL,
    pretrained_model_name NVARCHAR(100) NOT NULL,
    model_path NVARCHAR(255),
    hyperparameters NVARCHAR(MAX),
    feature_importances NVARCHAR(MAX),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    active BIT NOT NULL DEFAULT 0
);

-- Create TrainingHistory table
CREATE TABLE TrainingHistory (
    id NVARCHAR(36) PRIMARY KEY,
    model_config_id NVARCHAR(36) NOT NULL,
    metrics NVARCHAR(MAX) NOT NULL,
    train_date DATETIME NOT NULL,
    version NVARCHAR(50) NOT NULL,
    FOREIGN KEY (model_config_id) REFERENCES ModelConfigurations(id)
);

-- Create indexes to improve performance
CREATE INDEX IX_Keywords_DocumentTypeId ON Keywords(document_type_id);
CREATE INDEX IX_ClassificationRules_DocumentTypeId ON ClassificationRules(document_type_id);
CREATE INDEX IX_ClassificationRules_FeatureId ON ClassificationRules(feature_id);
CREATE INDEX IX_TrainingHistory_ModelConfigId ON TrainingHistory(model_config_id);
CREATE INDEX IX_ModelConfigurations_Active ON ModelConfigurations(active);