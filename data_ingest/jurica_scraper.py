#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scraper pour Jurica - Récupération des décisions de justice
"""

import os
import json
import jsonlines
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('jurica_scraper')

# Chemins des fichiers
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
os.makedirs(RAW_DIR, exist_ok=True)

# Configuration du scraper
BASE_URL = "https://www.cours-appel.justice.fr/jurica"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
}

def fetch_page(url: str) -> Optional[BeautifulSoup]:
    """
    Récupère une page web et retourne un objet BeautifulSoup.
    
    Args:
        url: L'URL de la page à récupérer
        
    Returns:
        Un objet BeautifulSoup ou None en cas d'erreur
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de la récupération de {url}: {e}")
        return None

def extract_decision_content(soup: BeautifulSoup, url: str) -> Dict:
    """
    Extrait le contenu d'une décision de justice.
    
    Args:
        soup: L'objet BeautifulSoup de la page de la décision
        url: L'URL de la décision
        
    Returns:
        Un dictionnaire contenant les informations de la décision
    """
    # Cette fonction doit être adaptée en fonction de la structure HTML de Jurica
    # Exemple simplifié:
    try:
        decision_id = url.split("/")[-1]
        title = soup.select_one("h1.title").text.strip()
        content = " ".join([p.text.strip() for p in soup.select("div.decision-content p")])
        date_text = soup.select_one("div.decision-date").text.strip()
        date = datetime.strptime(date_text, "%d/%m/%Y").strftime("%Y-%m-%d")
        juridiction = soup.select_one("div.juridiction").text.strip()
        
        return {
            "id": decision_id,
            "titre": title,
            "text": content,
            "date": date,
            "juridiction": juridiction,
            "source": "jurica",
            "url": url
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du contenu: {e}")
        return {}

def search_decisions(query: str, page: int = 1, max_results: int = 100) -> List[str]:
    """
    Recherche des décisions avec une requête spécifique.
    
    Args:
        query: La requête de recherche
        page: Le numéro de page
        max_results: Nombre maximum de résultats
        
    Returns:
        Une liste d'URLs vers les décisions trouvées
    """
    # URL de recherche à adapter selon l'API ou la structure de Jurica
    search_url = f"{BASE_URL}/search?query={query}&page={page}"
    decision_urls = []
    
    soup = fetch_page(search_url)
    if not soup:
        return decision_urls
    
    # Extraire les liens vers les décisions - à adapter en fonction de la structure réelle
    result_links = soup.select("div.search-results a.decision-link")
    
    for link in result_links:
        href = link.get("href")
        if href and len(decision_urls) < max_results:
            if not href.startswith("http"):
                href = BASE_URL + href
            decision_urls.append(href)
    
    return decision_urls

def scrape_jurica_decisions(query: str = "droit du travail", max_results: int = 100) -> List[Dict]:
    """
    Scrape les décisions de justice depuis Jurica.
    
    Args:
        query: Termes de recherche
        max_results: Nombre maximum de résultats
        
    Returns:
        Une liste de dictionnaires représentant les décisions
    """
    logger.info(f"Recherche de décisions avec la requête: {query}")
    
    decisions = []
    page = 1
    decision_urls = []
    
    # Récupérer les URLs des décisions
    while len(decision_urls) < max_results:
        new_urls = search_decisions(query, page, max_results - len(decision_urls))
        if not new_urls:
            break
        decision_urls.extend(new_urls)
        page += 1
    
    # Scraper chaque décision
    for url in tqdm(decision_urls):
        soup = fetch_page(url)
        if soup:
            decision_data = extract_decision_content(soup, url)
            if decision_data:
                decisions.append(decision_data)
    
    logger.info(f"{len(decisions)} décisions récupérées")
    return decisions

def save_to_jsonl(data: List[Dict], filename: str) -> None:
    """
    Sauvegarde les données au format JSONL.
    
    Args:
        data: Liste de dictionnaires à sauvegarder
        filename: Nom du fichier de sortie
    """
    output_path = RAW_DIR / filename
    with jsonlines.open(output_path, mode='w') as writer:
        for item in data:
            writer.write(item)
    logger.info(f"{len(data)} décisions sauvegardées dans {output_path}")

def main():
    """Point d'entrée principal du script."""
    logger.info("Démarrage du scraper Jurica")
    
    # Scrape des décisions sur le droit du travail
    travail_decisions = scrape_jurica_decisions(query="droit du travail", max_results=100)
    save_to_jsonl(travail_decisions, "jurica_travail.jsonl")
    
    # Scrape des décisions sur le droit civil
    civil_decisions = scrape_jurica_decisions(query="droit civil", max_results=100)
    save_to_jsonl(civil_decisions, "jurica_civil.jsonl")
    
    logger.info("Scraping terminé avec succès")

if __name__ == "__main__":
    main()
