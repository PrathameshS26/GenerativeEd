import streamlit as st
from streamlit_chat import message
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.title("AI Tutor For Higher Education")
st.subheader("Let's get started")

# Load the Transformer model and tokenizer
model_name = st.selectbox("Select a model", ("microsoft/DialoGPT-large", "google/mt5-small"))
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'chat_history_ids' not in st.session_state:
    st.session_state['chat_history_ids'] = None

query = st.text_input("Query: ", key="input")

if query:
    with st.spinner("Generating response..."):
        new_user_input_ids = tokenizer.encode(query + tokenizer.eos_token, return_tensors='pt')

        bot_input_ids = torch.cat([st.session_state['chat_history_ids'], new_user_input_ids], dim=-1) if st.session_state['chat_history_ids'] is not None else new_user_input_ids

        # Generate the response
        response_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.6
        )

        response_text = tokenizer.decode(response_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Update chat history
        st.session_state['chat_history_ids'] = bot_input_ids
        st.session_state.past.append(query)
        st.session_state.generated.append(response_text)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))

with st.expander("Show Messages"):
    st.write(st.session_state['chat_history_ids'])
