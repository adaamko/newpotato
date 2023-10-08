import requests
import streamlit as st

API_URL = "http://localhost:8000/"


def main():
    st.title("POTATO - HITL demo")

    # Initialize the session state
    if "state" not in st.session_state:
        st.session_state["state"] = "upload"

    # Upload a file
    if st.session_state["state"] == "upload":
        st.header("Upload a file")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            st.session_state["state"] = "classify"
            st.session_state["text"] = uploaded_file.read().decode("utf-8")

    # Classify
    elif st.session_state["state"] == "classify":
        st.header("Classifying text")
        if st.button("Parse Text"):
            try:
                with st.spinner("Parsing text..."):
                    response = requests.post(
                        API_URL + "parse",
                        json={"text": st.session_state["text"]},
                    )
                    response.raise_for_status()
                st.success("Text successfully parsed!")
                st.write(response.json())
            except Exception as e:
                st.error(f"An error occurred: {e}")

        if st.button("Go back to upload"):
            st.session_state["state"] = "upload"


if __name__ == "__main__":
    main()
