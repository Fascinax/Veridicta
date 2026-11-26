"""Load configured Streamlit secrets before initializing private resources."""

import os
from pathlib import Path

import streamlit as st
from streamlit import config


def load_server_config(keys):
    if not config.is_manually_set("secrets.files"):
        for path in st.get_option("secrets.files"):
            try:
                Path(path).lstat()
            except FileNotFoundError:
                continue
            break
        else:
            return
    values = {key: st.secrets.get(key) for key in keys}
    for key, value in values.items():
        if value and not os.getenv(key):
            os.environ[key] = str(value)
