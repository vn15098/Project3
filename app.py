from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from main import (
    BASE_DIR,
    analyze_query,
    create_vector_store,
    generate_answer,
)

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Banking Support Assistant",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 Banking Customer Support Assistant")
st.write("Ask a question about your account, transactions, or bank policies.")


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
    )


@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return create_vector_store(embeddings)


with st.form("query_form"):
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g., I saw an unfamiliar transaction on my account yesterday.",
    )
    submit_button = st.form_submit_button("Submit Query")

if submit_button and query.strip():
    with st.spinner("Processing your request..."):
        try:
            llm = get_llm()
            analysis = analyze_query(llm, query.strip())
            vector_store = get_vector_store()
            retrieved_documents = vector_store.as_retriever(
                search_kwargs={"k": 2}
            ).invoke(query.strip())
            answer_data = generate_answer(
                llm,
                query.strip(),
                analysis,
                retrieved_documents,
            )
            sources = list(
                dict.fromkeys(
                    document.metadata.get("source", "unknown")
                    for document in retrieved_documents
                )
            )

            st.markdown("---")
            st.subheader("Assistant Response")

            risk_level = answer_data["risk_level"]
            if risk_level:
                risk_color = {
                    "low": "🟢",
                    "medium": "🟡",
                    "high": "🔴",
                }.get(risk_level.lower(), "⚪")
                st.markdown(f"### Risk Level: {risk_color} {risk_level.upper()}")
            else:
                st.markdown("### Risk Level: ⚪ None / Not Assessed")

            st.markdown("### Suggested response")
            st.success(answer_data["response"])

            suggested_action = answer_data["suggested_action"]
            if suggested_action:
                st.info(f"💡 **Suggested Next Step:** {suggested_action}")

            if sources:
                st.caption(f"Sources: {', '.join(sources)}")

            with st.expander("View Pipeline Details"):
                st.write("**Intent Analysis:**", analysis)
                st.write("**Retrieved Documents Count:**", len(retrieved_documents))
                st.write("**Sources:**", sources)

        except Exception as error:
            st.error(f"An error occurred while processing your request: {error}")

elif submit_button:
    st.warning("Please enter a valid query before submitting.")
