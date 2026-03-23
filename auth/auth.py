import hashlib
import streamlit as st
from config.settings import SENHA_ADMIN, SENHA_SYN, SENHA_SME, SENHA_ENT


def make_hashes(password):
    if not password:
        return ""
    return hashlib.sha256(password.encode()).hexdigest()


def check_hashes(password, hashed_text):
    if not hashed_text:
        return False
    return make_hashes(password) == hashed_text


def _build_users():
    result = {}
    for username, senha in [
        ("admin", SENHA_ADMIN),
        ("SYN", SENHA_SYN),
        ("SME", SENHA_SME),
        ("Enterprise", SENHA_ENT),
    ]:
        if senha:
            result[username] = make_hashes(senha)
    return result


users = _build_users()


def get_current_user():
    return st.session_state.get('user', 'unknown')


def login():
    try:
        st.image('assets/macLogo.png', width=180)
    except Exception:
        st.markdown("# 🌱")
    st.title("Agente de Conteúdo")
    st.caption("Faça login para continuar")
    st.divider()

    with st.form("login_form"):
        username = st.text_input("Usuário", placeholder="Digite seu usuário")
        password = st.text_input("Senha", type="password", placeholder="Digite sua senha")
        submit_button = st.form_submit_button("Entrar", use_container_width=True)

        if submit_button:
            if username in users and check_hashes(password, users[username]):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Login realizado com sucesso!")
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos")


def check_admin_password():

    def admin_password_entered():
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
        st.error("😕 Senha de administrador incorreta")
        return False
    else:
        return True


def require_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()
        st.stop()


def logout():
    for key in ["logged_in", "user", "admin_password_correct", "admin_user"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
