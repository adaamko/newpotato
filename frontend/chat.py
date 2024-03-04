import streamlit as st
from openai import OpenAI


def chat(knowledge_base):
    client = OpenAI()

    prompt = st.chat_input("What is up?")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4"

    if "messages" not in st.session_state:
        system_prompt = """You are a helpful assistant who is answers questions about a knowledge base. You have access to triplets from the knowledge base and can use them to answer questions.
        You also have access to rules that were used to extract the triplets from raw text. You can use these rules to answer questions as well. For each triplet you also have information to the source sentences.
        Your job is to answer questions, retrieve knowledge or provide explanations. You can also ask for more information if you need it. You can also ask for help if you are stuck. 
        """

        kn_str = ""
        for item in knowledge_base:
            for sentence, applied_rules, triplets, _ in knowledge_base[item]:
                triplet = triplets[2]
                applied_rule = applied_rules[0]

                kn_str += f"Triplet: {triplet}\t Rule: {applied_rule}\t Sentence: {sentence}\n"
        system_prompt += f"\n\nKnowledge Base: {kn_str}"
        st.session_state.messages = [{"role": "system", "content": system_prompt}]

    for message in st.session_state.messages:
        if not message["role"] == "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
