#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traitement et normalisation des données juridiques brutes
"""

import os
import json
import jsonlines
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_processor')

# Chemins des fichiers
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Lit un fichier JSONL et retourne une liste de dictionnaires.
    
    Args:
        file_path: Chemin vers le fichier JSONL
        
    Returns:
        Une liste de dictionnaires contenus dans le fichier
    """
    data = []
    try:
        with jsonlines.open(file_path) as reader:
            for item in reader:
                data.append(item)
        logger.info(f"Lu {len(data)} éléments depuis {file_path}")
        return data
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier {file_path}: {e}")
        return []

def clean_text(text: str) -> str:
    """
    Nettoie et normalise un texte.
    
    Args:
        text: Le texte à nettoyer
        
    Returns:
        Le texte nettoyé
    """
    if not text:
        return ""
    
    # Supprimer les caractères de contrôle et les espaces multiples
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Normaliser les apostrophes et les guillemets
    text = text.replace("'", "'").replace('"', '"').replace('"', '"')
    
    return text.strip()

def standardize_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise un document juridique au format standard du projet.
    
    Args:
        doc: Le document source
        
    Returns:
        Le document normalisé
    """
    # Structure standard pour tous les documents
    standardized = {
        "id": doc.get("id", ""),
        "titre": clean_text(doc.get("titre", "")),
        "text": clean_text(doc.get("text", "")),
        "date": doc.get("date", ""),
        "source": doc.get("source", ""),
        "metadata": {}
    }
    
    # Ajouter des métadonnées spécifiques selon la source
    if doc.get("source") == "legifrance":
        standardized["metadata"]["code"] = doc.get("code", "")
        standardized["metadata"]["article_id"] = doc.get("article_id", "")
    elif doc.get("source") == "jurica":
        standardized["metadata"]["juridiction"] = doc.get("juridiction", "")
        standardized["metadata"]["type_decision"] = doc.get("type_decision", "")
    
    # Vérifier et corriger le format de date si nécessaire
    if standardized["date"]:
        try:
            # Convertir toutes les dates au format ISO
            date_obj = datetime.strptime(standardized["date"], "%Y-%m-%d")
            standardized["date"] = date_obj.strftime("%Y-%m-%d")
        except ValueError:
            try:
                # Essayer un autre format courant
                date_obj = datetime.strptime(standardized["date"], "%d/%m/%Y")
                standardized["date"] = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                logger.warning(f"Format de date invalide pour le document {standardized['id']}: {standardized['date']}")
                standardized["date"] = ""
    
    return standardized

def process_documents(input_files: List[Path], output_file: Path) -> None:
    """
    Traite plusieurs fichiers JSONL et les combine dans un seul fichier normalisé.
    
    Args:
        input_files: Liste des chemins vers les fichiers d'entrée
        output_file: Chemin du fichier de sortie
    """
    all_documents = []
    
    # Lire tous les documents d'entrée
    for file_path in input_files:
        if not file_path.exists():
            logger.warning(f"Le fichier {file_path} n'existe pas.")
            continue
        
        documents = read_jsonl(file_path)
        logger.info(f"Traitement de {len(documents)} documents depuis {file_path}")
        
        # Standardiser chaque document
        standardized_docs = []
        for doc in tqdm(documents, desc=f"Traitement {file_path.name}"):
            std_doc = standardize_document(doc)
            standardized_docs.append(std_doc)
        
        all_documents.extend(standardized_docs)
    
    # Supprimer les doublons en se basant sur l'ID
    unique_docs = {}
    for doc in all_documents:
        unique_docs[doc["id"]] = doc
    
    all_documents = list(unique_docs.values())
    logger.info(f"Total après déduplication: {len(all_documents)} documents")
    
    # Enregistrer les documents normalisés
    with jsonlines.open(output_file, mode='w') as writer:
        for doc in all_documents:
            writer.write(doc)
    
    logger.info(f"Documents normalisés enregistrés dans {output_file}")

def main():
    """Point d'entrée principal du script."""
    logger.info("Démarrage du traitement des données juridiques")
    
    # Fichiers à traiter
    input_files = [
        RAW_DIR / "code_travail.jsonl",
        RAW_DIR / "jurica_travail.jsonl",
        RAW_DIR / "jurica_civil.jsonl"
    ]
    
    # Traitement et fusion des documents
    process_documents(input_files, PROCESSED_DIR / "corpus_juridique.jsonl")
    
    logger.info("Traitement terminé avec succès")

if __name__ == "__main__":
    main()
