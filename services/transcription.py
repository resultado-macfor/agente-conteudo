"""
Serviço de transcrição de áudio e vídeo.
Utiliza a API do Gemini para transcrever arquivos de mídia.
"""
import google.generativeai as genai
from google.genai import types
from config.settings import GEMINI_API_KEY


def transcrever_audio_video(arquivo, tipo_arquivo: str) -> str:
    """
    Transcreve áudio ou vídeo usando a API do Gemini.

    Args:
        arquivo: Arquivo de áudio ou vídeo (objeto com .read() e .name)
        tipo_arquivo: "audio" ou "video"

    Returns:
        Texto transcrito ou mensagem de erro
    """
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Determina o tipo MIME
        extensao = arquivo.name.split('.')[-1].lower()
        if tipo_arquivo == "audio":
            mime_type = f"audio/{extensao}"
        else:
            mime_type = f"video/{extensao}"

        # Lê os bytes do arquivo
        arquivo_bytes = arquivo.read()

        # Para arquivos maiores que 20MB, usa upload
        if len(arquivo_bytes) > 20 * 1024 * 1024:
            uploaded_file = client.files.upload(file=arquivo_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=["Transcreva este arquivo em detalhes:", uploaded_file]
            )
        else:
            # Para arquivos menores, usa inline
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    "Transcreva este arquivo em detalhes:",
                    types.Part.from_bytes(data=arquivo_bytes, mime_type=mime_type)
                ]
            )

        return response.text

    except Exception as e:
        return f"Erro na transcrição: {str(e)}"
