"""
Sistema de autenticação.
Gerencia login, logout e verificação de usuários.
"""
import hashlib
import streamlit as st
from config.settings import SENHA_ADMIN, SENHA_SYN, SENHA_SME, SENHA_ENT


def make_hashes(password):
    """Gera hash SHA256 da senha."""
    if password is None:
        return None
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    """Verifica se a senha corresponde ao hash."""
    return make_hashes(password) == hashed_text


# Dados de usuário (em produção, isso deve vir de um banco de dados seguro)
USERS = {
    "admin": make_hashes(SENHA_ADMIN),
    "SYN": make_hashes(SENHA_SYN),
    "SME": make_hashes(SENHA_SME),
    "Enterprise": make_hashes(SENHA_ENT)
}


def get_current_user():
    """Retorna o usuário atual da sessão."""
    return st.session_state.get('user', 'unknown')


def login():
    """Formulário de login."""
    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if username in USERS and check_hashes(password, USERS[username]):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Login realizado com sucesso!")
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos")


def logout():
    """Realiza logout do usuário."""
    for key in ["logged_in", "user", "admin_password_correct", "admin_user"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def check_admin_password():
    """Retorna True se o usuário fornecer a senha de admin correta."""

    def admin_password_entered():
        """Verifica se a senha de admin está correta."""
        if st.session_state["admin_password"] == SENHA_ADMIN:
            st.session_state["admin_password_correct"] = True
            st.session_state["admin_user"] = "admin"
            del st.session_state["admin_password"]
        else:
            st.session_state["admin_password_correct"] = False

    if "admin_password_correct" not in st.session_state:
        st.text_input(
            "Senha de Administrador",
            type="password",
            on_change=admin_password_entered,
            key="admin_password"
        )
        return False
    elif not st.session_state["admin_password_correct"]:
        st.text_input(
            "Senha de Administrador",
            type="password",
            on_change=admin_password_entered,
            key="admin_password"
        )
        st.error("Senha de administrador incorreta")
        return False
    else:
        return True


def is_logged_in():
    """Verifica se o usuário está logado."""
    return st.session_state.get('logged_in', False)


def require_login():
    """Exige login para continuar. Retorna True se logado."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()
        st.stop()
        return False
    return True
