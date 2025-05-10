#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évaluation des performances du système RAG Veridicta
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm import tqdm

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluate')

# Chemins des fichiers
EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR.parent / "data"
RESULTS_DIR = EVAL_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@dataclass
class EvaluationResult:
    """Résultats d'une évaluation de système RAG"""
    model_name: str
    retriever_name: str
    exact_match_score: float
    f1_score: float
    retrieval_precision: float
    latency_ms: float
    token_count: int
    hallucination_rate: float
    cost_per_query: float

def load_evaluation_dataset(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Charge un dataset d'évaluation.
    
    Args:
        file_path: Chemin vers le fichier de dataset (json/jsonl)
        
    Returns:
        Liste des questions d'évaluation
    """
    # Si aucun fichier n'est spécifié, créer un dataset de test minimal
    if not file_path:
        return [
            {
                "id": "test-001",
                "question": "Quels sont les délais de préavis pour un licenciement économique?",
                "reference_answer": "Le délai de préavis pour un licenciement économique varie selon l'ancienneté: 1 mois pour une ancienneté de 6 mois à 2 ans, 2 mois pour une ancienneté de 2 ans ou plus.",
                "reference_sources": ["code_travail_L1234-1", "code_travail_L1234-2"]
            },
            {
                "id": "test-002",
                "question": "Quelles sont les conditions pour bénéficier du congé parental?",
                "reference_answer": "Pour bénéficier du congé parental, le salarié doit avoir au moins un an d'ancienneté dans l'entreprise à la date de naissance de l'enfant ou de l'adoption.",
                "reference_sources": ["code_travail_L1225-47", "code_travail_L1225-48"]
            }
        ]
    
    try:
        # Détecter le format du fichier
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            logger.error(f"Format de fichier non supporté: {file_path}")
            return []
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        return []

def calculate_exact_match(prediction: str, reference: str) -> float:
    """
    Calcule le score d'Exact Match entre la prédiction et la référence.
    
    Args:
        prediction: Réponse prédite
        reference: Réponse de référence
        
    Returns:
        Score entre 0 et 1
    """
    # Simplification pour l'instant - utiliser un matching plus sophistiqué plus tard
    norm_pred = prediction.lower().strip()
    norm_ref = reference.lower().strip()
    
    return 1.0 if norm_pred == norm_ref else 0.0

def calculate_f1(prediction: str, reference: str) -> float:
    """
    Calcule le score F1 entre la prédiction et la référence.
    
    Args:
        prediction: Réponse prédite
        reference: Réponse de référence
        
    Returns:
        Score F1 entre 0 et 1
    """
    # Tokenisation simple par mots
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    # Calcul du F1
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    common = len(pred_tokens.intersection(ref_tokens))
    precision = common / len(pred_tokens) if pred_tokens else 0
    recall = common / len(ref_tokens) if ref_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def calculate_hallucination_rate(prediction: str, reference_sources: List[str], retrieved_sources: List[Dict[str, Any]]) -> float:
    """
    Estime le taux d'hallucination en vérifiant si la prédiction se base sur des sources non récupérées.
    
    Args:
        prediction: Réponse prédite
        reference_sources: Sources de référence
        retrieved_sources: Sources récupérées
        
    Returns:
        Taux d'hallucination estimé entre 0 et 1
    """
    # Méthode simplifiée pour estimer les hallucinations
    # Une implémentation plus sophistiquée sera développée dans la phase 7
    
    # Si aucune source n'a été récupérée mais une réponse a été générée
    if not retrieved_sources and prediction.strip():
        return 1.0
    
    # Si au moins une source de référence a été récupérée
    retrieved_ids = [source.get("doc_id", "") for source in retrieved_sources]
    common_sources = set(retrieved_ids).intersection(set(reference_sources))
    
    if not common_sources:
        # Aucune source de référence n'a été récupérée
        return 0.8  # Haute probabilité d'hallucination
    
    # Ratio de sources pertinentes
    relevant_ratio = len(common_sources) / len(reference_sources)
    
    # Estimation du taux d'hallucination
    return max(0, 1 - relevant_ratio)

def evaluate_rag_system(retriever_module_path: str, retriever_class_name: str, 
                         model_name: str, questions: List[Dict[str, Any]]) -> EvaluationResult:
    """
    Évalue un système RAG sur un ensemble de questions.
    
    Args:
        retriever_module_path: Chemin d'importation du module retriever
        retriever_class_name: Nom de la classe du retriever
        model_name: Nom du modèle LLM
        questions: Liste des questions d'évaluation
        
    Returns:
        Résultat de l'évaluation
    """
    # Importer dynamiquement le module du retriever
    try:
        import importlib
        retriever_module = importlib.import_module(retriever_module_path)
        RetreiverClass = getattr(retriever_module, retriever_class_name)
        
        # Instancier le retriever
        retriever = RetreiverClass()
        
        # TODO: Configurer le modèle LLM dans la phase 2
        
        # Variables pour calculer les statistiques
        exact_match_scores = []
        f1_scores = []
        retrieval_precision_scores = []
        latency_ms_values = []
        token_counts = []
        hallucination_rates = []
        
        for question in tqdm(questions):
            # Mesurer le temps de réponse
            start_time = time.time()
            
            # Générer la réponse
            response, sources = retriever.generate_response(question["question"])
            
            # Calculer la latence
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculer les métriques
            em_score = calculate_exact_match(response, question["reference_answer"])
            f1 = calculate_f1(response, question["reference_answer"])
            
            # Précision du retrieval
            retrieved_ids = [source.get("doc_id", "") for source in sources]
            if question.get("reference_sources") and retrieved_ids:
                common = set(retrieved_ids).intersection(set(question["reference_sources"]))
                retrieval_precision = len(common) / len(retrieved_ids) if retrieved_ids else 0
            else:
                retrieval_precision = 0
            
            # Hallucinations
            hallucination_rate = calculate_hallucination_rate(
                response, question.get("reference_sources", []), sources
            )
            
            # Estimation du nombre de tokens (à remplacer par un comptage réel dans la phase 2)
            token_count = len(response.split()) + sum(len(source["text"].split()) for source in sources)
            
            # Stocker les résultats
            exact_match_scores.append(em_score)
            f1_scores.append(f1)
            retrieval_precision_scores.append(retrieval_precision)
            latency_ms_values.append(latency_ms)
            token_counts.append(token_count)
            hallucination_rates.append(hallucination_rate)
        
        # Calculer les moyennes
        avg_exact_match = np.mean(exact_match_scores) if exact_match_scores else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        avg_retrieval_precision = np.mean(retrieval_precision_scores) if retrieval_precision_scores else 0
        avg_latency_ms = np.mean(latency_ms_values) if latency_ms_values else 0
        avg_token_count = np.mean(token_counts) if token_counts else 0
        avg_hallucination_rate = np.mean(hallucination_rates) if hallucination_rates else 0
        
        # Estimer le coût moyen par requête (à affiner en phase 5)
        cost_per_1k_tokens = 0.005  # €/1k tokens, estimation initiale
        cost_per_query = (avg_token_count / 1000) * cost_per_1k_tokens
        
        return EvaluationResult(
            model_name=model_name,
            retriever_name=retriever_class_name,
            exact_match_score=avg_exact_match,
            f1_score=avg_f1,
            retrieval_precision=avg_retrieval_precision,
            latency_ms=avg_latency_ms,
            token_count=int(avg_token_count),
            hallucination_rate=avg_hallucination_rate,
            cost_per_query=cost_per_query
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {e}")
        # Retourner un résultat vide en cas d'erreur
        return EvaluationResult(
            model_name=model_name,
            retriever_name=retriever_class_name,
            exact_match_score=0.0,
            f1_score=0.0,
            retrieval_precision=0.0,
            latency_ms=0.0,
            token_count=0,
            hallucination_rate=1.0,
            cost_per_query=0.0
        )

def main():
    """Point d'entrée principal du script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Évaluation du système RAG Veridicta")
    parser.add_argument("--model", type=str, default="placeholder",
                        help="Modèle LLM à évaluer (placeholder, mistral7B, mistral7B_qlora)")
    parser.add_argument("--retriever", type=str, default="baseline",
                        help="Retriever à évaluer (baseline, light_rag, path_rag)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Chemin vers le dataset d'évaluation")
    
    args = parser.parse_args()
    
    logger.info(f"Évaluation du système avec le modèle '{args.model}' et le retriever '{args.retriever}'")
    
    # Mapper les noms aux classes/modules
    retriever_mapping = {
        "baseline": ("retrievers.baseline_rag", "BaselineRAG"),
        "light_rag": ("retrievers.light_rag_retriever", "LightRAG"),
        "path_rag": ("retrievers.path_rag_retriever", "PathRAG"),
    }
    
    if args.retriever not in retriever_mapping:
        logger.error(f"Retriever '{args.retriever}' non supporté")
        return
    
    # Charger le dataset
    questions = load_evaluation_dataset(args.dataset)
    logger.info(f"Évaluation sur {len(questions)} questions")
    
    # Évaluer le système
    retriever_module_path, retriever_class_name = retriever_mapping[args.retriever]
    result = evaluate_rag_system(
        retriever_module_path=retriever_module_path,
        retriever_class_name=retriever_class_name,
        model_name=args.model,
        questions=questions
    )
    
    # Afficher les résultats
    logger.info("Résultats de l'évaluation:")
    logger.info(f"- Modèle: {result.model_name}")
    logger.info(f"- Retriever: {result.retriever_name}")
    logger.info(f"- Score Exact Match: {result.exact_match_score:.2%}")
    logger.info(f"- Score F1: {result.f1_score:.2%}")
    logger.info(f"- Précision du retrieval: {result.retrieval_precision:.2%}")
    logger.info(f"- Latence moyenne: {result.latency_ms:.2f} ms")
    logger.info(f"- Nombre moyen de tokens: {result.token_count}")
    logger.info(f"- Taux d'hallucination estimé: {result.hallucination_rate:.2%}")
    logger.info(f"- Coût moyen par requête: {result.cost_per_query:.6f} €")
    
    # Sauvegarder les résultats
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = RESULTS_DIR / f"eval_{args.model}_{args.retriever}_{timestamp}.json"
    with open(result_file, "w") as f:
        # Convertir le dataclass en dict
        result_dict = {k: v for k, v in result.__dict__.items()}
        json.dump(result_dict, f, indent=2)
    
    logger.info(f"Résultats sauvegardés dans {result_file}")

if __name__ == "__main__":
    main()
