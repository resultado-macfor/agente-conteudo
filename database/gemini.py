"""
Configuração dos modelos Gemini.
"""
import google.generativeai as genai
from config.settings import (
    GEMINI_API_KEY,
    MODELO_VISION,
    MODELO_TEXTO,
    MODELO_TEXTO_PRO
)

# =============================================================================
# CONFIGURAÇÃO DO GEMINI
# =============================================================================
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    modelo_vision = genai.GenerativeModel(MODELO_VISION, generation_config={"temperature": 0.0})
    modelo_texto = genai.GenerativeModel(MODELO_TEXTO)
    modelo_texto2 = genai.GenerativeModel(MODELO_TEXTO_PRO)
else:
    modelo_vision = None
    modelo_texto = None
    modelo_texto2 = None
