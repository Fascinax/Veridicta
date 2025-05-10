#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototype de base pour le système RAG (Retrieval Augmented Generation)
"""

import os
import json
import jsonlines
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('baseline_rag')

# Chemins des fichiers
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Configuration du modèle d'embeddings
MODEL_NAME = "distiluse-base-multilingual-cased-v1"  # À remplacer par MiniLM-FR plus tard

class BaselineRAG:
    """Implémentation de base du système RAG pour le droit français"""
    
    def __init__(self, corpus_path: Path = PROCESSED_DIR / "corpus_juridique.jsonl",
                 embeddings_path: Path = EMBEDDINGS_DIR / "faiss_index",
                 model_name: str = MODEL_NAME):
        """
        Initialise le système RAG.
        
        Args:
            corpus_path: Chemin vers le corpus juridique
            embeddings_path: Chemin pour sauvegarder/charger les embeddings
            model_name: Nom du modèle SentenceTransformer à utiliser
        """
        self.corpus_path = corpus_path
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        
        # Chargement du modèle d'embeddings
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Index FAISS et documents
        self.index = None
        self.documents = []
        self.doc_ids = []
        
        # Template de prompt RAG
        self.rag_template = """
        Tâche: En tant qu'assistant juridique spécialisé dans le droit français, réponds à la question de l'utilisateur en utilisant uniquement les informations des articles et décisions fournis ci-dessous. 
        Ne fais pas de supposition et ne cite pas de textes juridiques qui ne seraient pas inclus dans les sources.

        Sources:
        {sources}
        
        Question: {query}
        
        Réponds de manière claire, concise et précise en te basant uniquement sur les sources fournies. 
        Cite les références exactes (numéros d'articles, décisions) entre parenthèses pour justifier ta réponse.
        Si les sources ne contiennent pas suffisamment d'informations pour répondre, indique-le honnêtement.
        
        Réponse:
        """
        self.rag_prompt = PromptTemplate.from_template(self.rag_template)
    
    def load_documents(self) -> None:
        """Charge les documents du corpus"""
        if not self.corpus_path.exists():
            logger.error(f"Le corpus n'existe pas: {self.corpus_path}")
            return
        
        try:
            with jsonlines.open(self.corpus_path) as reader:
                self.documents = []
                self.doc_ids = []
                
                for doc in reader:
                    # Nous devons créer des chunks pour que le retrieval soit plus précis
                    chunks = self._split_document(doc)
                    self.documents.extend(chunks)
                    self.doc_ids.extend([doc["id"]] * len(chunks))
            
            logger.info(f"Chargé {len(self.documents)} chunks depuis {self.corpus_path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du corpus: {e}")
    
    def _split_document(self, doc: Dict[str, Any]) -> List[str]:
        """
        Divise un document en chunks pour un meilleur retrieval.
        
        Args:
            doc: Document à diviser
            
        Returns:
            Liste des chunks de texte
        """
        # Méthode simple pour diviser en paragraphes
        text = doc["text"]
        title = doc["titre"]
        
        # Ajouter le titre comme premier chunk
        chunks = [title]
        
        # Diviser en paragraphes (approximatif)
        paragraphs = text.split('\n\n')
        chunks.extend([p.strip() for p in paragraphs if p.strip()])
        
        # Si le texte est court, le considérer comme un seul chunk
        if len(chunks) <= 1:
            return [text]
        
        return chunks
    
    def build_index(self) -> None:
        """Construit l'index FAISS à partir des documents chargés"""
        if not self.documents:
            logger.warning("Aucun document chargé. Veuillez d'abord charger le corpus.")
            return
        
        logger.info("Calcul des embeddings...")
        embeddings = []
        
        # Calculer les embeddings par batch pour optimiser
        batch_size = 32
        for i in tqdm(range(0, len(self.documents), batch_size)):
            batch = self.documents[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            embeddings.extend(batch_embeddings)
        
        # Convertir en numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Créer l'index FAISS
        logger.info("Création de l'index FAISS...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings_array)
        
        # Sauvegarder l'index et les métadonnées
        logger.info(f"Sauvegarde de l'index dans {self.embeddings_path}...")
        os.makedirs(self.embeddings_path.parent, exist_ok=True)
        faiss.write_index(self.index, str(self.embeddings_path) + ".index")
        
        # Sauvegarder les documents et IDs
        with open(str(self.embeddings_path) + ".json", "w") as f:
            json.dump({
                "documents": self.documents,
                "doc_ids": self.doc_ids
            }, f)
        
        logger.info("Index FAISS construit et sauvegardé avec succès.")
    
    def load_index(self) -> bool:
        """
        Charge l'index FAISS et les documents associés.
        
        Returns:
            True si le chargement a réussi, False sinon
        """
        index_path = str(self.embeddings_path) + ".index"
        docs_path = str(self.embeddings_path) + ".json"
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            logger.warning(f"L'index ou les documents n'existent pas: {self.embeddings_path}")
            return False
        
        try:
            # Charger l'index
            self.index = faiss.read_index(index_path)
            
            # Charger les documents
            with open(docs_path, "r") as f:
                data = json.load(f)
                self.documents = data["documents"]
                self.doc_ids = data["doc_ids"]
            
            logger.info(f"Index chargé avec {self.index.ntotal} vecteurs et {len(self.documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'index: {e}")
            return False
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Récupère les documents les plus pertinents pour une requête.
        
        Args:
            query: La question ou requête
            k: Nombre de documents à récupérer
            
        Returns:
            Liste des documents les plus pertinents
        """
        if self.index is None:
            logger.error("L'index n'est pas chargé")
            return []
        
        # Calculer l'embedding de la requête
        query_embedding = self.embedding_model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Rechercher les plus proches voisins
        distances, indices = self.index.search(query_embedding, k)
        
        # Récupérer les documents correspondants
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "doc_id": self.doc_ids[idx],
                    "score": float(distances[0][i])
                })
        
        return results
    
    def generate_response(self, query: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Génère une réponse à une question en utilisant le RAG.
        
        Args:
            query: La question posée
            k: Nombre de documents à récupérer
            
        Returns:
            Tuple contenant la réponse générée et les sources utilisées
        """
        # TODO: Dans la phase suivante, cette fonction sera complétée avec LLM
        sources = self.retrieve(query, k)
        
        if not sources:
            return "Je n'ai pas pu accéder aux sources juridiques nécessaires pour répondre à cette question.", []
        
        # Format des sources pour le prompt
        formatted_sources = "\n\n".join([f"Source {i+1}: {source['text']}" for i, source in enumerate(sources)])
        
        # TODO: Intégrer un LLM ici pour générer la réponse finale
        # Pour l'instant, retourne un message de placeholder
        response = f"À implémenter dans la phase 2: intégration du LLM pour générer une réponse basée sur les sources suivantes: {', '.join([s['doc_id'] for s in sources])}"
        
        return response, sources

def main():
    """Point d'entrée principal du script."""
    logger.info("Initialisation du système RAG de base")
    
    rag = BaselineRAG()
    
    # Vérifier si l'index existe
    if not rag.load_index():
        logger.info("Construction d'un nouvel index...")
        rag.load_documents()
        rag.build_index()
    
    # Test simple avec une requête
    query = "Quelles sont les indemnités légales de licenciement?"
    logger.info(f"Test avec la requête: '{query}'")
    
    response, sources = rag.generate_response(query)
    
    logger.info(f"Réponse: {response}")
    logger.info(f"Sources: {len(sources)} documents récupérés")

if __name__ == "__main__":
    main()
