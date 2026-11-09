"""Stop unauthorized sessions before opening private storage or RAG resources."""

import os
import uuid

import streamlit as st

from workspaces.access import AccessDenied, authorized_workspace_owner
from workspaces.store import WorkspaceStore


def _clear_private_session() -> None:
    for key in list(st.session_state):
        if key != "workspace_session_id":
            del st.session_state[key]


def get_workspace_store() -> WorkspaceStore:
    mode = os.getenv("VERIDICTA_WORKSPACE_MODE", "session")
    if "workspace_session_id" not in st.session_state:
        st.session_state.workspace_session_id = uuid.uuid4().hex
    try:
        owner = authorized_workspace_owner(mode, st.session_state.workspace_session_id, st.user)
    except AccessDenied:
        _clear_private_session()
        st.error("Ce compte n'est pas autorisé à utiliser l'application. Contactez l'administrateur.")
        if st.button("Se déconnecter"):
            st.logout()
        st.stop()
    except PermissionError:
        _clear_private_session()
        if st.user.get("is_logged_in") is True:
            st.info("Votre session a expiré ou votre identité est invalide. Reconnectez-vous.")
            if st.button("Se reconnecter"):
                st.logout()
        else:
            st.info("Connectez-vous pour accéder à vos dossiers.")
            if st.button("Se connecter"):
                st.login()
        st.stop()
    except ValueError:
        _clear_private_session()
        st.error("Le contrôle d'accès est mal configuré. Contactez l'administrateur.")
        st.stop()
    if mode == "session":
        st.sidebar.caption("Dossiers limités à cette session. Exportez vos mémos avant de la fermer.")
    elif mode == "local":
        st.sidebar.caption("Mode personnel : stockage local commun aux sessions de ce serveur.")
    elif st.sidebar.button("Se déconnecter"):
        _clear_private_session()
        st.logout()
        st.stop()
    store = WorkspaceStore(os.getenv("VERIDICTA_WORKSPACE_DIR", "data/cases"), owner_id=owner)
    previous_scope = st.session_state.get("workspace_owner_scope")
    if previous_scope is not None and previous_scope != store.scope_key:
        _clear_private_session()
    st.session_state.workspace_owner_scope = store.scope_key
    return store
