#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scraper pour le Journal de Monaco - Récupération des bulletins officiels
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import re

import requests
from bs4 import BeautifulSoup
import jsonlines
from tqdm import tqdm
from playwright.sync_api import sync_playwright

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('monaco_scraper')

# Chemins des fichiers
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
MONACO_DIR = RAW_DIR / "monaco"
os.makedirs(MONACO_DIR, exist_ok=True)
os.makedirs(MONACO_DIR / "pdfs", exist_ok=True)

# Configuration du scraper
BASE_URL = "https://journaldemonaco.gouv.mc"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
}

def check_robots_txt(url: str = BASE_URL) -> bool:
    """
    Vérifie que le scraping est autorisé par le robots.txt.
    
    Args:
        url: URL de base du site à scraper
        
    Returns:
        True si le scraping est autorisé, False sinon
    """
    try:
        robots_url = f"{url}/robots.txt" if url[-1] != "/" else f"{url}robots.txt"
        response = requests.get(robots_url, timeout=10)
        
        if response.status_code == 200:
            content = response.text
            # Vérification simple: si "Disallow: /" est présent, le scraping est interdit
            if "Disallow: /" in content and "Allow: /" not in content:
                logger.warning("Le robots.txt interdit le scraping global")
                return False
            logger.info("Le scraping semble autorisé par le robots.txt")
            return True
        else:
            logger.warning(f"Impossible de récupérer robots.txt ({response.status_code})")
            return True  # Par défaut, on autorise si on ne peut pas vérifier
            
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du robots.txt: {e}")
        return True  # Par défaut, on autorise si on ne peut pas vérifier

def get_journal_list(max_pages: int = 5) -> List[Dict]:
    """
    Récupère la liste des journaux disponibles en utilisant Playwright pour le JavaScript.
    
    Args:
        max_pages: Nombre maximum de pages à récupérer
        
    Returns:
        Liste des journaux trouvés
    """
    journals = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        
        # Aller sur la page d'index
        page.goto(f"{BASE_URL}/tous-les-journaux", timeout=60000)
        page.wait_for_selector(".journal-item")
        
        current_page = 1
        
        while current_page <= max_pages:
            # Extraire les journaux de la page courante
            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            
            journal_items = soup.select(".journal-item")
            if not journal_items:
                break
                
            logger.info(f"Page {current_page}: {len(journal_items)} journaux trouvés")
            
            for item in journal_items:
                try:
                    title_elem = item.select_one(".journal-titre")
                    date_elem = item.select_one(".journal-date")
                    link_elem = item.select_one("a.journal-pdf")
                    
                    if title_elem and date_elem and link_elem:
                        title = title_elem.text.strip()
                        date_text = date_elem.text.strip()
                        pdf_url = link_elem.get("href")
                        
                        # Extraire le numéro
                        numero_match = re.search(r"n°\s*(\d+)", title)
                        numero = numero_match.group(1) if numero_match else "unknown"
                        
                        # Formater la date
                        try:
                            date_obj = datetime.strptime(date_text, "%d/%m/%Y")
                            formatted_date = date_obj.strftime("%Y-%m-%d")
                        except:
                            formatted_date = date_text
                        
                        journal = {
                            "numero": numero,
                            "titre": title,
                            "date": formatted_date,
                            "pdf_url": f"{BASE_URL}{pdf_url}" if pdf_url.startswith("/") else pdf_url,
                            "id": f"JM_{numero}_{formatted_date.replace('-', '')}"
                        }
                        journals.append(journal)
                except Exception as e:
                    logger.error(f"Erreur lors de l'extraction d'un journal: {e}")
            
            # Passer à la page suivante si disponible
            next_button = page.query_selector(".pagination .next:not(.disabled)")
            if not next_button:
                break
                
            next_button.click()
            page.wait_for_timeout(2000)  # Attendre le chargement
            current_page += 1
        
        browser.close()
    
    return journals

def download_pdf(url: str, output_path: Path) -> bool:
    """
    Télécharge un PDF depuis une URL.
    
    Args:
        url: URL du PDF
        output_path: Chemin où sauvegarder le PDF
        
    Returns:
        True si le téléchargement a réussi, False sinon
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
            
        logger.info(f"PDF téléchargé: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement de {url}: {e}")
        return False

def process_journals(journals: List[Dict], download_pdfs: bool = True) -> None:
    """
    Traite les journaux récupérés et télécharge les PDFs si demandé.
    
    Args:
        journals: Liste des journaux
        download_pdfs: Si True, télécharge les PDFs
    """
    # Sauvegarder les métadonnées
    metadata_path = MONACO_DIR / "journaux_monaco.jsonl"
    
    with jsonlines.open(metadata_path, mode='w') as writer:
        for journal in journals:
            writer.write(journal)
    
    logger.info(f"{len(journals)} métadonnées de journaux sauvegardées dans {metadata_path}")
    
    # Télécharger les PDFs si demandé
    if download_pdfs:
        for journal in tqdm(journals, desc="Téléchargement des PDFs"):
            pdf_filename = f"{journal['id']}.pdf"
            pdf_path = MONACO_DIR / "pdfs" / pdf_filename
            
            # Vérifier si le PDF existe déjà
            if pdf_path.exists():
                logger.debug(f"PDF déjà existant: {pdf_path}")
                continue
                
            # Télécharger le PDF
            download_pdf(journal["pdf_url"], pdf_path)

def main():
    """Point d'entrée principal du script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scraper pour le Journal de Monaco")
    parser.add_argument("--max-pages", type=int, default=5, help="Nombre maximum de pages à récupérer")
    parser.add_argument("--no-download", action="store_true", help="Ne pas télécharger les PDFs")
    
    args = parser.parse_args()
    
    logger.info("Démarrage du scraper pour le Journal de Monaco")
    
    # Vérifier le robots.txt
    if not check_robots_txt():
        logger.warning("Le scraping pourrait être interdit par le robots.txt. Continuer ?")
        response = input("Continuer quand même ? (o/n): ").lower()
        if response != 'o':
            logger.info("Arrêt du script")
            return
    
    # Récupérer la liste des journaux
    journals = get_journal_list(max_pages=args.max_pages)
    
    if not journals:
        logger.error("Aucun journal trouvé")
        return
        
    logger.info(f"{len(journals)} journaux trouvés")
    
    # Traiter les journaux et télécharger les PDFs si demandé
    process_journals(journals, download_pdfs=not args.no_download)
    
    logger.info("Scraping terminé avec succès")

if __name__ == "__main__":
    main()
