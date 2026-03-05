"""
A ideia é facilitar e obter informações mais rapidamente a cerca de algum vídeo no Youtube, aí tive a ideia de criar um assistente feito em IA, 
inicialmente eu cogitei o modelo de IA da OpenIA, entretanto ocorreu um conflito entre poder e querer... No caso da OpenIA, após gerar a Chave API, 
é necessário ter um saldo de pelo menos 5 dólares para poder utilizar o modelo de IA, ChatGPT versão 4. 
Com base no custo e com a ideia de minimizar gaastos, optei pelo modelo de IA da Gemini (gemini-2.5-flash") que é totalmente gratuito.
Bom, para criar a minha ideia procurei alguns vídeos sobre assistentes de IA, e com base em alguns, e analisando uma IA de análise de Viagens, modelei
um analista de URLs de Vídeos do Youtube.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import YoutubeLoader

template = """Você é um assistente de análise de URLs, focado em vídeos do YouTube. 
Você recebe a transcrição do vídeo e gera um resumo detalhado sobre o conteúdo.
Na sequência, pergunte ao usuário o que ele deseja aprofundar sobre o vídeo.

Histórico da conversa:
{history}

Entrada do usuário ou Transcrição do vídeo:
{input}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="history"), 
    ("human", "{input}") 
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

chain = prompt | llm 

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:  
        store[session_id] = ChatMessageHistory()
    return store[session_id] 

chat_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def iniciar_assistente_analise():
    print("Bem vindo ao seu analista de URLs! Digite 'sair' para encerrar.\n") 
    while True:
        pergunta_user = input("Você: ")
        
        if pergunta_user.lower() in ["sair", "exit"]: 
            print("Assistente de URL: Até mais!")
            break

        texto_para_ia = pergunta_user

        if "youtube.com" in pergunta_user or "youtu.be" in pergunta_user:
            print("Assistente de URL: Hum, vi que é um vídeo do YouTube! Baixando as legendas, um momento...")
            try:
                # O loader extrai o texto do vídeo, a coleta de dados é feita a partir da leitura das legendas
                loader = YoutubeLoader.from_youtube_url(pergunta_user, language=["pt", "en"])
                documentos = loader.load()
                transcricao = documentos[0].page_content
                
                texto_para_ia = f"Aqui está a transcrição do vídeo contido nesta URL ({pergunta_user}):\n\n{transcricao}\n\nPor favor, faça o resumo."
            except Exception as e:
                print(f"Assistente de URL: Ops! Não consegui puxar as legendas desse vídeo. Motivo: {e}")
                continue
        
        # Envia para o Gemini
        resposta = chat_with_history.invoke(
            {'input' : texto_para_ia},
            config={'configurable' : {'session_id': 'user123'}}
        )
        
        print(f"\nAssistente de URL: {resposta.content}")
        print("-" * 50) 
        
if __name__ == "__main__": 
    iniciar_assistente_analise()