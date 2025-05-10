#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration du graphe de connaissances Neo4j pour LightRAG
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from neo4j import GraphDatabase
from dotenv import load_dotenv

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('neo4j_setup')

# Charger les variables d'environnement
load_dotenv()

# Configuration Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class Neo4jManager:
    """Gestionnaire de la base de données Neo4j pour le graphe de connaissances juridiques"""
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        """
        Initialise le gestionnaire Neo4j.
        
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
    
    def clear_database(self) -> None:
        """Supprime toutes les données et contraintes de la base de données"""
        if not self.driver:
            logger.error("Non connecté à Neo4j")
            return
        
        with self.driver.session() as session:
            # Supprimer toutes les contraintes et indexes
            session.run("CALL apoc.schema.assert({}, {})")
            # Supprimer toutes les données
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Base de données Neo4j réinitialisée")
    
    def create_schema(self) -> None:
        """Crée le schéma initial pour le graphe de connaissances juridiques"""
        if not self.driver:
            logger.error("Non connecté à Neo4j")
            return
        
        with self.driver.session() as session:
            # Créer les contraintes pour les nœuds
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Code) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            
            # Créer des index pour améliorer les performances
            session.run("CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.date)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Decision) ON (d.date)")
            
            logger.info("Schéma Neo4j créé avec succès")
    
    def add_article(self, article: Dict[str, Any]) -> bool:
        """
        Ajoute un article au graphe.
        
        Args:
            article: Données de l'article
            
        Returns:
            True si l'ajout a réussi, False sinon
        """
        if not self.driver:
            logger.error("Non connecté à Neo4j")
            return False
        
        try:
            with self.driver.session() as session:
                # Créer le nœud Article
                result = session.run(
                    """
                    MERGE (a:Article {id: $id})
                    ON CREATE SET 
                        a.titre = $titre,
                        a.text = $text,
                        a.date = $date
                    RETURN a
                    """,
                    id=article["id"],
                    titre=article["titre"],
                    text=article["text"],
                    date=article["date"]
                )
                
                # Vérifier si l'article a été créé
                record = result.single()
                if not record:
                    logger.warning(f"L'article {article['id']} n'a pas pu être créé")
                    return False
                
                # Si l'article appartient à un code, créer la relation
                if "code" in article.get("metadata", {}):
                    code_id = article["metadata"]["code"]
                    session.run(
                        """
                        MERGE (c:Code {id: $code_id})
                        MERGE (a:Article {id: $article_id})
                        MERGE (a)-[:BELONGS_TO]->(c)
                        """,
                        code_id=code_id,
                        article_id=article["id"]
                    )
                
                # Ajouter des thèmes si présents
                if "themes" in article.get("metadata", {}):
                    for theme in article["metadata"]["themes"]:
                        session.run(
                            """
                            MERGE (t:Theme {name: $theme})
                            MERGE (a:Article {id: $article_id})
                            MERGE (a)-[:HAS_THEME]->(t)
                            """,
                            theme=theme,
                            article_id=article["id"]
                        )
                
                logger.debug(f"Article {article['id']} ajouté avec succès")
                return True
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de l'article {article['id']}: {e}")
            return False
    
    def add_decision(self, decision: Dict[str, Any]) -> bool:
        """
        Ajoute une décision de justice au graphe.
        
        Args:
            decision: Données de la décision
            
        Returns:
            True si l'ajout a réussi, False sinon
        """
        if not self.driver:
            logger.error("Non connecté à Neo4j")
            return False
        
        try:
            with self.driver.session() as session:
                # Créer le nœud Decision
                result = session.run(
                    """
                    MERGE (d:Decision {id: $id})
                    ON CREATE SET 
                        d.titre = $titre,
                        d.text = $text,
                        d.date = $date,
                        d.juridiction = $juridiction
                    RETURN d
                    """,
                    id=decision["id"],
                    titre=decision["titre"],
                    text=decision["text"],
                    date=decision["date"],
                    juridiction=decision.get("metadata", {}).get("juridiction", "")
                )
                
                # Vérifier si la décision a été créée
                record = result.single()
                if not record:
                    logger.warning(f"La décision {decision['id']} n'a pas pu être créée")
                    return False
                
                # Créer des relations avec les articles de loi cités
                if "references" in decision.get("metadata", {}):
                    for article_id in decision["metadata"]["references"]:
                        session.run(
                            """
                            MERGE (a:Article {id: $article_id})
                            MERGE (d:Decision {id: $decision_id})
                            MERGE (d)-[:CITES]->(a)
                            """,
                            article_id=article_id,
                            decision_id=decision["id"]
                        )
                
                # Ajouter des thèmes si présents
                if "themes" in decision.get("metadata", {}):
                    for theme in decision["metadata"]["themes"]:
                        session.run(
                            """
                            MERGE (t:Theme {name: $theme})
                            MERGE (d:Decision {id: $decision_id})
                            MERGE (d)-[:HAS_THEME]->(t)
                            """,
                            theme=theme,
                            decision_id=decision["id"]
                        )
                
                logger.debug(f"Décision {decision['id']} ajoutée avec succès")
                return True
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de la décision {decision['id']}: {e}")
            return False

def main():
    """Point d'entrée principal du script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration de Neo4j pour Veridicta")
    parser.add_argument("--reset", action="store_true", help="Réinitialiser la base de données")
    parser.add_argument("--input", type=str, help="Fichier JSONL contenant les documents à charger")
    
    args = parser.parse_args()
    
    # Connexion à Neo4j
    neo4j = Neo4jManager()
    if not neo4j.connect():
        logger.error("Impossible de se connecter à Neo4j. Vérifiez les paramètres de connexion.")
        return
    
    try:
        # Réinitialiser la base de données si demandé
        if args.reset:
            logger.info("Réinitialisation de la base de données Neo4j...")
            neo4j.clear_database()
        
        # Créer le schéma
        logger.info("Création du schéma Neo4j...")
        neo4j.create_schema()
        
        # Charger des documents si un fichier est spécifié
        if args.input and os.path.exists(args.input):
            logger.info(f"Chargement des documents depuis {args.input}...")
            
            import jsonlines
            total = 0
            success = 0
            
            with jsonlines.open(args.input) as reader:
                for doc in reader:
                    total += 1
                    source = doc.get("source", "").lower()
                    
                    if source == "legifrance":
                        if neo4j.add_article(doc):
                            success += 1
                    elif source in ["jurica", "judilibre"]:
                        if neo4j.add_decision(doc):
                            success += 1
                    else:
                        logger.warning(f"Source inconnue: {source}")
            
            logger.info(f"Chargement terminé: {success}/{total} documents importés avec succès")
    
    finally:
        # Fermer la connexion
        neo4j.close()

if __name__ == "__main__":
    main()
