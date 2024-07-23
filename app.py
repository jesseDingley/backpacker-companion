import streamlit as st

from backend.core import get_hello_world

st.title(get_hello_world())