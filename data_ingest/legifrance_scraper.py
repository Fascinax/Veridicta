#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scraper pour Legifrance - Récupération des textes juridiques
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
logger = logging.getLogger('legifrance_scraper')

# Chemins des fichiers
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
os.makedirs(RAW_DIR, exist_ok=True)

# Configuration du scraper
BASE_URL = "https://www.legifrance.gouv.fr"
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

def extract_article_content(soup: BeautifulSoup) -> Dict:
    """
    Extrait le contenu d'un article juridique.
    
    Args:
        soup: L'objet BeautifulSoup de la page de l'article
        
    Returns:
        Un dictionnaire contenant les informations de l'article
    """
    # Cette fonction doit être adaptée en fonction de la structure HTML de Legifrance
    # Exemple simplifié:
    try:
        article_id = soup.select_one("article").get("id", "")
        title = soup.select_one("h1.title").text.strip()
        content = " ".join([p.text.strip() for p in soup.select("div.article-content p")])
        date_text = soup.select_one("div.article-date").text.strip()
        date = datetime.strptime(date_text, "%d/%m/%Y").strftime("%Y-%m-%d")
        
        return {
            "id": article_id,
            "titre": title,
            "text": content,
            "date": date,
            "source": "legifrance",
            "url": soup.url
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du contenu: {e}")
        return {}

def scrape_code_du_travail() -> List[Dict]:
    """
    Scrape le Code du Travail depuis Legifrance.
    
    Returns:
        Une liste de dictionnaires représentant les articles
    """
    logger.info("Scraping du Code du Travail...")
    
    # Note: Cette URL et cette logique doivent être adaptées à la structure réelle de Legifrance
    code_url = f"{BASE_URL}/codes/code.do?cidTexte=LEGITEXT000006072050"
    articles = []
    
    # Logique simplifiée - à adapter:
    soup = fetch_page(code_url)
    if not soup:
        return articles
    
    # Récupérer les liens vers les articles
    article_links = soup.select("a.articleLink")
    
    for link in tqdm(article_links[:10]):  # Limité à 10 articles pour le test
        href = link.get("href")
        if href:
            article_url = BASE_URL + href
            article_soup = fetch_page(article_url)
            if article_soup:
                article_data = extract_article_content(article_soup)
                if article_data:
                    articles.append(article_data)
    
    logger.info(f"{len(articles)} articles récupérés du Code du Travail")
    return articles

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
    logger.info(f"{len(data)} articles sauvegardés dans {output_path}")

def main():
    """Point d'entrée principal du script."""
    logger.info("Démarrage du scraper Legifrance")
    
    # Scrape du Code du Travail
    code_travail_articles = scrape_code_du_travail()
    save_to_jsonl(code_travail_articles, "code_travail.jsonl")
    
    logger.info("Scraping terminé avec succès")

if __name__ == "__main__":
    main()
