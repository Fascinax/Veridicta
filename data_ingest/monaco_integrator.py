#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intégrateur des sources juridiques monégasques dans Neo4j
"""

import os
import json
import jsonlines
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('monaco_integrator')

# Charger les variables d'environnement
load_dotenv()

# Chemins des fichiers
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MONACO_DIR = DATA_DIR / "raw" / "monaco"

# Configuration Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "veridicta2025")

# Modèles d'extraction d'entités
ENTITY_PATTERNS = {
    "loi": r"Loi\s+n°\s+(\d+[\d\-\.\/]*)",
    "ordonnance": r"Ordonnance\s+Souveraine\s+n°\s+(\d+[\d\-\.\/]*)",
    "arrete": r"Arrêté\s+Ministériel\s+n°\s+(\d+[\d\-\.\/]*)",
    "projet": r"Projet\s+de\s+loi\s+n°\s+(\d+[\d\-\.\/]*)"
}

class MonacoIntegrator:
    """Classe pour intégrer les sources juridiques monégasques dans Neo4j"""
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        """
        Initialise l'intégrateur.
        
        Args:
            uri: URI de connexion à Neo4j
            user: Nom d'utilisateur
            password: Mot de passe
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
    def connect(self) -> bool:
        """
        Établit une connexion à la base de données Neo4j.
        
        Returns:
            True si la connexion a réussi, False sinon
        """
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Vérifier la connexion
            self.driver.verify_connectivity()
            logger.info("Connexion à Neo4j établie avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur de connexion à Neo4j: {e}")
            return False
    
    def close(self) -> None:
        """Ferme la connexion à la base de données"""
        if self.driver:
            self.driver.close()
            logger.info("Connexion à Neo4j fermée")

    def setup_monaco_schema(self) -> None:
        """Crée le schéma pour les sources monégasques dans Neo4j"""
        if not self.driver:
            logger.error("Non connecté à Neo4j")
            return
            
        with self.driver.session() as session:
            # Créer les contraintes pour les nœuds
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (j:JournalMonaco) REQUIRE j.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:CodeMonegasque) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:ArticleMonegasque) REQUIRE a.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:DecisionMonaco) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:LoiMonaco) REQUIRE l.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (o:OrdonnanceMonaco) REQUIRE o.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:ArreteMonaco) REQUIRE a.id IS UNIQUE")
            
            # Créer des index pour améliorer les performances
            session.run("CREATE INDEX IF NOT EXISTS FOR (j:JournalMonaco) ON (j.date)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (a:ArticleMonegasque) ON (a.date_vigueur)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:DecisionMonaco) ON (d.date)")
            
            logger.info("Schéma Monaco créé avec succès dans Neo4j")

    def add_journal_monaco(self, journal: Dict[str, Any]) -> bool:
        """
        Ajoute un journal de Monaco au graphe.
        
        Args:
            journal: Données du journal
            
        Returns:
            True si l'ajout a réussi, False sinon
        """
        if not self.driver:
            logger.error("Non connecté à Neo4j")
            return False
            
        try:
            with self.driver.session() as session:
                # Créer le nœud JournalMonaco
                result = session.run(
                    """
                    MERGE (j:JournalMonaco {id: $id})
                    ON CREATE SET 
                        j.numero = $numero,
                        j.titre = $titre,
                        j.date = $date,
                        j.pdf_url = $pdf_url,
                        j.source = 'Journal de Monaco'
                    RETURN j
                    """,
                    id=journal["id"],
                    numero=journal["numero"],
                    titre=journal["titre"],
                    date=journal["date"],
                    pdf_url=journal["pdf_url"]
                )
                
                # Vérifier si le journal a été créé
                record = result.single()
                if not record:
                    logger.warning(f"Le journal {journal['id']} n'a pas pu être créé")
                    return False
                    
                logger.debug(f"Journal {journal['id']} ajouté avec succès")
                return True
                
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du journal {journal['id']}: {e}")
            return False
    
    def add_code_monaco(self, code: Dict[str, Any]) -> bool:
        """
        Ajoute un code monégasque au graphe.
        
        Args:
            code: Données du code
            
        Returns:
            True si l'ajout a réussi, False sinon
        """
        if not self.driver:
            logger.error("Non connecté à Neo4j")
            return False
            
        try:
            with self.driver.session() as session:
                # Créer le nœud CodeMonegasque
                result = session.run(
                    """
                    MERGE (c:CodeMonegasque {id: $id})
                    ON CREATE SET 
                        c.titre = $titre,
                        c.date_version = $date_version,
                        c.source = 'LegiMonaco'
                    RETURN c
                    """,
                    id=code["id"],
                    titre=code["titre"],
                    date_version=code.get("date_version", "")
                )
                
                # Vérifier si le code a été créé
                record = result.single()
                if not record:
                    logger.warning(f"Le code {code['id']} n'a pas pu être créé")
                    return False
                
                # Ajouter des articles si présents
                if "articles" in code:
                    for article in code["articles"]:
                        self.add_article_monaco(article, code["id"])
                
                logger.debug(f"Code {code['id']} ajouté avec succès")
                return True
                
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du code {code['id']}: {e}")
            return False

    def add_article_monaco(self, article: Dict[str, Any], code_id: str = None) -> bool:
        """
        Ajoute un article monégasque au graphe.
        
        Args:
            article: Données de l'article
            code_id: ID du code parent (optionnel)
            
        Returns:
            True si l'ajout a réussi, False sinon
        """
        if not self.driver:
            logger.error("Non connecté à Neo4j")
            return False
            
        try:
            with self.driver.session() as session:
                # Créer le nœud ArticleMonegasque
                result = session.run(
                    """
                    MERGE (a:ArticleMonegasque {id: $id})
                    ON CREATE SET 
                        a.numero = $numero,
                        a.texte = $texte,
                        a.date_vigueur = $date_vigueur,
                        a.source = 'LegiMonaco'
                    RETURN a
                    """,
                    id=article["id"],
                    numero=article.get("numero", ""),
                    texte=article["texte"],
                    date_vigueur=article.get("date_vigueur", "")
                )
                
                # Vérifier si l'article a été créé
                record = result.single()
                if not record:
                    logger.warning(f"L'article {article['id']} n'a pas pu être créé")
                    return False
                
                # Relier l'article à son code si spécifié
                if code_id:
                    session.run(
                        """
                        MATCH (a:ArticleMonegasque {id: $article_id})
                        MATCH (c:CodeMonegasque {id: $code_id})
                        MERGE (a)-[:BELONGS_TO]->(c)
                        """,
                        article_id=article["id"],
                        code_id=code_id
                    )
                
                # Extraire les références à d'autres textes
                if "texte" in article:
                    self.extract_and_link_references(article["id"], article["texte"])
                
                logger.debug(f"Article {article['id']} ajouté avec succès")
                return True
                
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de l'article {article['id']}: {e}")
            return False
    
    def extract_and_link_references(self, source_id: str, text: str) -> None:
        """
        Extrait les références à des textes juridiques et crée les liens dans Neo4j.
        
        Args:
            source_id: ID du document source
            text: Texte à analyser
        """
        if not self.driver:
            logger.error("Non connecté à Neo4j")
            return
            
        try:
            with self.driver.session() as session:
                # Extraire les références aux lois
                loi_refs = re.finditer(ENTITY_PATTERNS["loi"], text)
                for match in loi_refs:
                    loi_num = match.group(1)
                    loi_id = f"loi_monaco_{loi_num}"
                    
                    # Créer le nœud Loi s'il n'existe pas
                    session.run(
                        """
                        MERGE (l:LoiMonaco {id: $id})
                        ON CREATE SET l.numero = $numero
                        """,
                        id=loi_id,
                        numero=loi_num
                    )
                    
                    # Créer le lien CITES
                    session.run(
                        """
                        MATCH (s {id: $source_id})
                        MATCH (l:LoiMonaco {id: $loi_id})
                        MERGE (s)-[:CITES]->(l)
                        """,
                        source_id=source_id,
                        loi_id=loi_id
                    )
                
                # De même pour les ordonnances, arrêtés, etc.
                # Code similaire pour chaque type de référence...
                
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des références: {e}")
    
    def import_journal_monaco_data(self, file_path: Path) -> None:
        """
        Importe les données des journaux de Monaco depuis un fichier JSONL.
        
        Args:
            file_path: Chemin vers le fichier JSONL
        """
        if not file_path.exists():
            logger.error(f"Le fichier {file_path} n'existe pas")
            return
            
        try:
            journals_count = 0
            success_count = 0
            
            with jsonlines.open(file_path) as reader:
                for journal in tqdm(reader, desc="Importation des journaux de Monaco"):
                    journals_count += 1
                    if self.add_journal_monaco(journal):
                        success_count += 1
            
            logger.info(f"Importation terminée: {success_count}/{journals_count} journaux importés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'importation des journaux: {e}")

def main():
    """Point d'entrée principal du script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intégration des sources juridiques monégasques dans Neo4j")
    parser.add_argument("--journals", type=str, help="Chemin vers le fichier JSONL des journaux de Monaco")
    
    args = parser.parse_args()
    
    integrator = MonacoIntegrator()
    
    if not integrator.connect():
        logger.error("Impossible de se connecter à Neo4j. Vérifiez les paramètres de connexion.")
        return
    
    try:
        # Configurer le schéma
        integrator.setup_monaco_schema()
        
        # Importer les données des journaux si spécifié
        if args.journals and os.path.exists(args.journals):
            integrator.import_journal_monaco_data(Path(args.journals))
        elif args.journals:
            logger.error(f"Le fichier {args.journals} n'existe pas")
        else:
            # Utiliser le chemin par défaut si aucun n'est spécifié
            default_path = MONACO_DIR / "journaux_monaco.jsonl"
            if default_path.exists():
                logger.info(f"Utilisation du fichier par défaut: {default_path}")
                integrator.import_journal_monaco_data(default_path)
            else:
                logger.warning(f"Aucun fichier de journaux trouvé à {default_path}")
                
    finally:
        # Fermer la connexion
        integrator.close()

if __name__ == "__main__":
    main()
