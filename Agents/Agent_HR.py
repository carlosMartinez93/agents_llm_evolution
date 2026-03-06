# ============================================
# PROJETO 3 — AGENTE DE RH COM RAG + RERANKING
# LangChain + Streamlit
# ============================================

# =========================
# 1. IMPORTAÇÕES
# =========================

import os
import streamlit as st

# Injeta a chave como variável de ambiente

# Loaders e chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings e LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Vector Store
from langchain_community.vectorstores import Chroma

# Prompt
from langchain_core.prompts import PromptTemplate


# =========================
# 2. CONFIGURAÇÕES GERAIS
# =========================

# Diretório do banco vetorial
PERSIST_DIRECTORY = "./chroma_rh"

# Modelo de embeddings
EMBEDDING_MODEL = "text-embedding-3-small"

# Modelo de linguagem
LLM_MODEL = "gpt-4o-mini"

# =========================
# 3. LEITURA DOS DOCUMENTOS
# =========================

@st.cache_data
def carregar_documentos():
    """
    Carrega os PDFs de políticas internas de RH
    """
    caminhos = [
        "politica_ferias.pdf",
        "politica_home_office.pdf",
        "codigo_conduta.pdf"
    ]

    documentos = []

    for caminho in caminhos:
        loader = PyPDFLoader(caminho)
        docs = loader.load()

        for doc in docs:
            doc.metadata["documento"] = caminho

        documentos.extend(docs)

    return documentos

# =========================
# 4. CHUNKING
# =========================

def gerar_chunks(documentos):
    """
    Divide os documentos em chunks semânticos
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(documentos)

# =========================
# 5. ENRIQUECIMENTO COM METADADOS
# =========================

def enriquecer_chunks(chunks):
    """
    Classifica os chunks por categoria semântica
    """
    for chunk in chunks:
        texto = chunk.page_content.lower()

        if "férias" in texto:
            chunk.metadata["categoria"] = "ferias"
        elif "home office" in texto or "remoto" in texto:
            chunk.metadata["categoria"] = "home_office"
        elif "conduta" in texto or "ética" in texto:
            chunk.metadata["categoria"] = "conduta"
        else:
            chunk.metadata["categoria"] = "geral"

    return chunks

# =========================
# 6. VECTOR STORE
# =========================

@st.cache_resource
def criar_vectorstore(_chunks):
    """
    Cria ou carrega o banco vetorial.
    O parâmetro _chunks não entra no hash do cache.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    return vectorstore

# =========================
# 7. RERANKING (PARTE CHAVE!)
# =========================

def rerank_documentos(pergunta, documentos, llm):
    """
    Reordena os documentos recuperados com base na relevância
    usando o próprio LLM (reranking semântico)
    """

    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template="""
Você é um especialista em políticas internas de RH.

Pergunta do usuário:
{pergunta}

Trecho do documento:
{texto}

Avalie a relevância desse trecho para responder a pergunta.
Responda apenas com um número de 0 a 10.
"""
    )

    documentos_com_score = []

    for doc in documentos:
        score = llm.invoke(
            prompt_rerank.format(
                pergunta=pergunta,
                texto=doc.page_content
            )
        ).content

        try:
            score = float(score)
        except:
            score = 0

        documentos_com_score.append((score, doc))

    # Ordena do mais relevante para o menos relevante
    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda x: x[0],
        reverse=True
    )

    # Retorna apenas os documentos
    return [doc for _, doc in documentos_ordenados]

# =========================
# 8. PIPELINE RAG COMPLETO
# =========================

def responder_pergunta(pergunta, vectorstore):
    """
    Pipeline completo:
    - Recuperação
    - Reranking
    - Geração de resposta
    """

    # LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    # Recuperação inicial (top-k mais alto)
    documentos_recuperados = vectorstore.similarity_search(
        pergunta,
        k=8
    )

    # Reranking
    documentos_rerankeados = rerank_documentos(
        pergunta,
        documentos_recuperados,
        llm
    )

    # Seleciona os melhores
    contexto_final = documentos_rerankeados[:4]

    # Prompt final
    contexto_texto = "\n\n".join(
        [doc.page_content for doc in contexto_final]
    )

    prompt_final = f"""
Você é um agente de RH corporativo.
Responda APENAS com base nas políticas internas abaixo.

Contexto:
{contexto_texto}

Pergunta:
{pergunta}
"""

    resposta = llm.invoke(prompt_final)

    return resposta.content, contexto_final

# =========================
# 9. INTERFACE STREAMLIT
# =========================

st.set_page_config(page_title="Agente de RH com RAG", layout="wide")
st.title("🤖 Agente de RH — Políticas Internas")

pergunta = st.text_input("Digite sua pergunta sobre políticas internas de RH:")

if pergunta:
    with st.spinner("Consultando políticas internas..."):
        documentos = carregar_documentos()
        chunks = gerar_chunks(documentos)
        chunks = enriquecer_chunks(chunks)
        vectorstore = criar_vectorstore(chunks)

        resposta, fontes = responder_pergunta(pergunta, vectorstore)

    st.subheader("Resposta")
    st.write(resposta)

    st.subheader("Fontes utilizadas")
    for i, doc in enumerate(fontes, start=1):
        st.markdown(f"**Trecho {i}**")
        st.write(f"Documento: {doc.metadata.get('documento')}")
        st.write(f"Categoria: {doc.metadata.get('categoria')}")
        st.write(doc.page_content)
        st.divider()


## Quais são as regras para concessão de férias aos colaboradores?

## Quem pode trabalhar em regime de home office e quais são as condições?

## Quais comportamentos são considerados inadequados segundo o código de conduta da empresa?
