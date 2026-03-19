# ============================================
# PROJETO 3 — AGENTE DE RH COM RAG + RERANKING
# LangChain + Streamlit
# LM Studio (Gemma) + Embeddings Locais
# ============================================

# =========================
# 1. IMPORTAÇÕES
# =========================

import os
import streamlit as st

# Loaders e chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LLM via LM Studio (OpenAI-compatible)
from langchain_openai import ChatOpenAI

# Vector Store (use o wrapper novo)
from langchain_chroma import Chroma

# Prompt
from langchain_core.prompts import PromptTemplate

# Embeddings locais (sem OpenAI / sem LM Studio)
from sentence_transformers import SentenceTransformer

# Para reconstruir Document após query manual no Chroma
from langchain_core.documents import Document


# =========================
# 2. CONFIGURAÇÕES GERAIS
# =========================

# Diretório do banco vetorial
PERSIST_DIRECTORY = "./chroma_rh"

# Embeddings locais (modelo leve e bom)
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims

# LLM local (LM Studio)
LLM_MODEL = "google/gemma-3-1b"
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"


# =========================
# 3. LEITURA DOS DOCUMENTOS
# =========================

@st.cache_data
def carregar_documentos():
    """
    Carrega os PDFs de políticas internas de RH
    """
    caminhos = [
        "C:\Users\Administrador\OneDrive\Documentos\Profissional\Alura\2_Engenharia_IA\Arq_RAG\PDFs\Office policies\codigo_conduta.pdf",
        "C:\Users\Administrador\OneDrive\Documentos\Profissional\Alura\2_Engenharia_IA\Arq_RAG\PDFs\Office policies\politica_home_office.pdf",
        "C:\Users\Administrador\OneDrive\Documentos\Profissional\Alura\2_Engenharia_IA\Arq_RAG\PDFs\Office policies\codigo_conduta.pdf"
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
# 6. VECTOR STORE (EMBEDDINGS LOCAIS)
# =========================

@st.cache_resource
def criar_vectorstore(_chunks):
    """
    Cria ou carrega o banco vetorial.
    Indexa usando embeddings locais (sentence-transformers).
    """
    embedder = SentenceTransformer(LOCAL_EMBED_MODEL)

    texts = [d.page_content for d in _chunks]
    metadatas = [d.metadata for d in _chunks]

    vectors = embedder.encode(texts, normalize_embeddings=True).tolist()

    vectorstore = Chroma(
        collection_name="rh_policies",
        persist_directory=PERSIST_DIRECTORY
    )

    # Adiciona tudo (IDs estáveis por índice)
    vectorstore._collection.add(
        ids=[str(i) for i in range(len(texts))],
        documents=texts,
        metadatas=metadatas,
        embeddings=vectors
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
            score = float(score.strip())
        except:
            score = 0.0

        documentos_com_score.append((score, doc))

    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in documentos_ordenados]


# =========================
# 8. PIPELINE RAG COMPLETO
# =========================

def responder_pergunta(pergunta, vectorstore):
    """
    Pipeline completo:
    - Recuperação (query manual no Chroma)
    - Reranking
    - Geração de resposta
    """

    # LLM local via LM Studio
    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LMSTUDIO_BASE_URL,
        api_key="lm-studio",  # qualquer string
        temperature=0
    )

    # Recuperação inicial usando embeddings locais (mesmo do índice)
    embedder = SentenceTransformer(LOCAL_EMBED_MODEL)
    qvec = embedder.encode(pergunta, normalize_embeddings=True).tolist()

    res = vectorstore._collection.query(
        query_embeddings=[qvec],
        n_results=8,
        include=["documents", "metadatas"]
    )

    documentos_recuperados = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(res["documents"][0], res["metadatas"][0])
    ]

    # Reranking
    documentos_rerankeados = rerank_documentos(
        pergunta,
        documentos_recuperados,
        llm
    )

    # Seleciona os melhores
    contexto_final = documentos_rerankeados[:4]

    # Prompt final
    contexto_texto = "\n\n".join([doc.page_content for doc in contexto_final])

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
st.title("🤖 Agente de RH — Políticas Internas (LM Studio)")

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


# Exemplos:
# Quais são as regras para concessão de férias aos colaboradores?
# Quem pode trabalhar em regime de home office e quais são as condições?
# Quais comportamentos são considerados inadequados segundo o código de conduta da empresa?