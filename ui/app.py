#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface utilisateur Streamlit pour Veridicta
"""

import os
import sys
import logging
from pathlib import Path

import streamlit as st
from streamlit.logger import get_logger

# Ajouter le répertoire parent au PYTHONPATH
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Configuration du logger
logger = get_logger(__name__)

# Importer notre système RAG (à adapter selon l'avancement du projet)
try:
    from retrievers.baseline_rag import BaselineRAG
    rag_system = BaselineRAG()
    has_rag = True
except ImportError:
    logger.warning("Système RAG non disponible, mode démo activé")
    has_rag = False

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Veridicta - Assistant Juridique IA",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

def format_sources(sources):
    """Formate les sources pour l'affichage"""
    if not sources:
        return ""
    
    source_text = "**Sources:**\n\n"
    for i, source in enumerate(sources):
        doc_id = source.get("doc_id", "inconnu")
        text = source.get("text", "...")
        score = source.get("score", 0)
        
        source_text += f"{i+1}. **{doc_id}** (score: {score:.2f})\n"
        source_text += f"   {text[:200]}{'...' if len(text) > 200 else ''}\n\n"
    
    return source_text

def main():
    """Fonction principale de l'application Streamlit"""
    # Titre et description
    st.title("⚖️ Veridicta")
    st.subheader("Assistant juridique français basé sur l'IA")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2091/2091722.png", width=100)
        st.markdown("### À propos")
        st.info(
            "Veridicta est un assistant juridique spécialisé dans le droit français. "
            "Il utilise des technologies RAG (Retrieval-Augmented Generation) pour fournir "
            "des réponses précises et traçables basées sur les textes juridiques et la jurisprudence."
        )
        
        st.markdown("### Paramètres")
        retriever = st.selectbox(
            "Système de recherche",
            options=["BaselineRAG", "LightRAG", "PathRAG"],
            index=0,
            disabled=not has_rag,
        )
        
        model = st.selectbox(
            "Modèle LLM",
            options=["Mistral-7B", "Mistral-7B (QLoRA)", "Mixtral-8x7B"],
            index=0,
            disabled=True,  # Sera activé dans une phase ultérieure
        )
        
        show_sources = st.checkbox("Afficher les sources", value=True)
        
        st.markdown("### Métriques")
        if has_rag:
            st.metric("Temps de réponse moyen", "0.8s")
            st.metric("Précision", "~50%")
        else:
            st.warning("Mode démo - Métriques non disponibles")
    
    # Zone de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Afficher les messages précédents
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and show_sources and "sources" in message:
                with st.expander("Afficher les sources"):
                    st.markdown(message["sources"])
    
    # Zone de saisie pour la question
    if prompt := st.chat_input("Posez votre question juridique..."):
        # Ajouter la question à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Afficher la question
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Préparer la réponse
        with st.chat_message("assistant"):
            if has_rag:
                with st.spinner("Recherche en cours..."):
                    try:
                        # Appeler le système RAG
                        response, sources = rag_system.generate_response(prompt)
                        formatted_sources = format_sources(sources)
                    except Exception as e:
                        response = f"Désolé, une erreur s'est produite: {str(e)}"
                        formatted_sources = ""
            else:
                # Mode démo - réponse factice
                response = (
                    "Mode démo: Le système RAG n'est pas encore disponible. "
                    "Cet assistant sera capable de répondre à vos questions juridiques "
                    "en se basant sur les textes de loi et la jurisprudence française."
                )
                formatted_sources = ""
            
            # Afficher la réponse
            st.markdown(response)
            
            if show_sources and formatted_sources:
                with st.expander("Afficher les sources"):
                    st.markdown(formatted_sources)
            
            # Ajouter la réponse à l'historique
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": formatted_sources
            })

if __name__ == "__main__":
    main()
