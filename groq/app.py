import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import time
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("ChatGroq With GPT Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions:{input}
    """
)


if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "final_documents" not in st.session_state:
    st.session_state.final_documents = None

def format_response_to_dict(response, question):
    """
    Robust formatter that returns:
      {"question", "tool_used", "tool_result", "final_answer"}
    Handles: response as dict-with-messages, dict-with-answer, plain string, or other.
    """
    tool_used = None
    tool_result = None
    final_answer = None

    # 1) If response is a plain string -> treat as final answer
    if isinstance(response, str):
        final_answer = response
        return {
            "question": question,
            "tool_used": tool_used,
            "tool_result": tool_result,
            "final_answer": final_answer,
        }

    # 2) If response is a dict-like object with explicit top-level answer/output keys
    if isinstance(response, dict):
        # Common LCEL parser output: {"answer": "..."} or {"output": "..."}
        if "answer" in response or "output" in response or "result" in response:
            final_answer = response.get("answer") or response.get("output") or response.get("result")
            # Try to extract tool info from 'messages' or 'context' if present
            messages = response.get("messages") or response.get("message") or None
            # fall through to messages handling below if present
            if not messages:
                return {
                    "question": question,
                    "tool_used": tool_used,
                    "tool_result": tool_result,
                    "final_answer": final_answer,
                }
        else:
            # maybe the dict contains 'messages' key that we need to inspect
            messages = response.get("messages", None)
            if messages is None:
                # nothing useful found — stringify fallback
                final_answer = str(response)
                return {
                    "question": question,
                    "tool_used": tool_used,
                    "tool_result": tool_result,
                    "final_answer": final_answer,
                }
    else:
        # 3) If response is an object (not dict or str), try to access response["messages"] via dict protocol
        messages = None
        try:
            messages = response["messages"]
        except Exception:
            messages = None

    # At this point we hope to have `messages` as a sequence (list/tuple)
    if isinstance(messages, (list, tuple)):
        for m in messages:
            # support both object messages (HumanMessage / AIMessage / ToolMessage) and dict messages
            # Identify message role/type
            m_type = None
            # object-style (has attributes)
            if hasattr(m, "type"):
                m_type = getattr(m, "type", None)
            elif isinstance(m, dict):
                # dict-style may use 'role' or 'type'
                m_type = m.get("type") or m.get("role")
                # normalize values like 'human'/'assistant' to 'human'/'ai' etc if needed
                if isinstance(m_type, str) and m_type.lower() == "assistant":
                    m_type = "ai"

            # TOOL CALL detection inside assistant/ai messages
            # object-style: has attribute tool_calls
            if m_type == "ai":
                if hasattr(m, "tool_calls") and getattr(m, "tool_calls"):
                    try:
                        tc = m.tool_calls[0]
                        tool_used = tc.get("name") if isinstance(tc, dict) else tc["name"]
                    except Exception:
                        pass
                elif isinstance(m, dict) and m.get("tool_calls"):
                    tc = m["tool_calls"][0]
                    tool_used = tc.get("name")

            # TOOL MESSAGE: get tool output
            if m_type == "tool":
                # object-style: m.content, m.name
                if hasattr(m, "content"):
                    tool_result = getattr(m, "content")
                elif isinstance(m, dict):
                    tool_result = m.get("content")
                # also try to get tool name if not known
                if not tool_used:
                    if hasattr(m, "name"):
                        tool_used = getattr(m, "name")
                    elif isinstance(m, dict):
                        tool_used = m.get("name")

            # Final assistant answer: ai message without tool_calls
            if m_type == "ai":
                has_tool_calls = False
                if hasattr(m, "tool_calls") and getattr(m, "tool_calls"):
                    has_tool_calls = True
                if isinstance(m, dict) and m.get("tool_calls"):
                    has_tool_calls = True
                if not has_tool_calls:
                    # extract content
                    if hasattr(m, "content"):
                        final_answer = getattr(m, "content")
                    elif isinstance(m, dict):
                        # dict-style assistant message may store content under 'content' or 'text'
                        final_answer = m.get("content") or m.get("text")
                    # continue scanning in case later messages override

    else:
        # As a fallback, stringify the response
        final_answer = str(response)

    return {
        "question": question,
        "tool_used": tool_used,
        "tool_result": tool_result,
        "final_answer": final_answer,
    }


def vector_embeddings():
    """Build vector store and stash it into session state."""
    with st.spinner("Building vector store — this may take a bit..."):
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.success("Vector Store DB is ready")

prompt1 = st.text_input("Enter Your Question from Documents")

if st.button("Documents Embeddings"):
    vector_embeddings()

# Retrieval Chain Helpers
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

format_docs_runnable = RunnableLambda(format_docs)

# Only create and run a retrieval chain if vectors exist
if st.session_state.vectors is None:
    st.info("Vector DB not ready. Click 'Documents Embeddings' to build it first.")
else:
    # Build retriever *now* because vectors exist
    raw_retriever = st.session_state.vectors.as_retriever()

    # ---- SAFETY WRAPPER: ensure retriever ALWAYS receives a plain string query ----
    # We keep the variable name `retriever` so the rest of your code is unchanged.
    def _extract_query(x):
        # If it's already a string -> return it
        if isinstance(x, str):
            return x
        # If it's a dict like {"input": "..."} or {"query":"..."} -> pick sensible keys
        if isinstance(x, dict):
            for k in ("input", "query", "text", "q"):
                if k in x and isinstance(x[k], str):
                    return x[k]
            # if dict contains a nested value that is string, try to find it
            for v in x.values():
                if isinstance(v, str):
                    return v
        # If it is a list/tuple with a single string element
        if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], str):
            return x[0]
        # Fallback: coerce to string
        return str(x)

    def _get_docs_safe(q):
        query = _extract_query(q)
        # Try common retriever APIs robustly:
        if hasattr(raw_retriever, "get_relevant_documents"):
            return raw_retriever.get_relevant_documents(query)
        if hasattr(raw_retriever, "retrieve"):
            return raw_retriever.retrieve(query)
        if hasattr(raw_retriever, "invoke"):
            out = raw_retriever.invoke(query)
            if isinstance(out, dict) and "documents" in out:
                return out["documents"]
            return out
        # fallback: if retriever is callable
        if callable(raw_retriever):
            out = raw_retriever(query)
            if isinstance(out, dict) and "documents" in out:
                return out["documents"]
            return out
        # nothing matched — raise a clear error
        raise AttributeError("Retriever object has no known retrieval method (get_relevant_documents/retrieve/invoke).")

    # Wrap into RunnableLambda so LCEL uses it safely
    retriever = RunnableLambda(_get_docs_safe)

    # Build LCEL retrieval chain dynamically (safe)
    document_chain = prompt | llm | StrOutputParser()
    retrieval_chain = (
        {"context": retriever | format_docs_runnable, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if prompt1:
        start = time.process_time()
        with st.spinner("Querying..."):
            response = retrieval_chain.invoke({"input": prompt1})
            clean = format_response_to_dict(response, prompt1)
            st.write(clean)
        st.write("Response time (s):", time.process_time() - start)

        # response shape from StrOutputParser may be {'answer': '...'} or plain string depending on your parser
        # try to be tolerant:
        answer = None
        if isinstance(response, dict) and "answer" in response:
            answer = response["answer"]
        elif isinstance(response, dict) and "output" in response:
            answer = response["output"]
        else:
            # fallback: try the object itself as string
            answer = str(response)

        st.write(answer)

        with st.expander("Document Similarity Search"):
                # First try to see if LCEL returned context
            ctx = response.get("context") if isinstance(response, dict) else None

            if ctx:
                docs_for_display = ctx
            else:
                # Fallback: directly use the retriever to get actual relevant documents
                try:
                    docs_for_display = raw_retriever.get_relevant_documents(prompt1)[:10]
                except AttributeError:
                    # fallback API names
                    if hasattr(raw_retriever, "retrieve"):
                        docs_for_display = raw_retriever.retrieve(prompt1)[:10]
                    elif hasattr(raw_retriever, "invoke"):
                        out = raw_retriever.invoke(prompt1)
                        if isinstance(out, dict) and "documents" in out:
                            docs_for_display = out["documents"][:10]
                        else:
                            docs_for_display = out[:10]
                    else:
                        docs_for_display = []

            if not docs_for_display:
                st.write("No retrieved documents to display.")
            else:
                for i, doc in enumerate(docs_for_display):
                    # doc might be Document or dict — support both
                    text = getattr(doc, "page_content", None) or \
                        (doc.get("page_content") if isinstance(doc, dict) else str(doc))

                    src = None
                    if hasattr(doc, "metadata"):
                        src = doc.metadata.get("source")
                    elif isinstance(doc, dict):
                        src = doc.get("source")

                    if src:
                        st.write(f"**Source:** {src}")

                    st.write(text)
                    st.write("---")

