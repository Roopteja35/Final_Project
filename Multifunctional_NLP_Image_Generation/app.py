import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusionPipeline
import torch

st.title("Multifunctional AI Tool")

# -------------------------------
# Caching model loaders for performance
# -------------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

@st.cache_resource
def load_text_generator():
    return pipeline("text-generation", model="gpt2")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_qa():
    return pipeline("question-answering")

@st.cache_resource
def load_image_generator():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32  # Use float32 for CPU
    )
    pipe.enable_attention_slicing()
    return pipe

@st.cache_resource
def load_dialo_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

# -------------------------------
# UI and Task Selector
# -------------------------------
task = st.sidebar.selectbox("Choose a Task", [
    "Text Summarization", "Next Word Prediction", "Story Generation", 
    "Chatbot", "Sentiment Analysis", "Question Answering", "Image Generation"
])

# -------------------------------
# Task Logic
# -------------------------------
if task == "Text Summarization":
    summarizer = load_summarizer()
    text = st.text_area("Enter text to summarize")
    if st.button("Summarize"):
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        st.write("Summary:", summary[0]['summary_text'])

elif task == "Next Word Prediction":
    generator = load_text_generator()
    input_text = st.text_input("Start your sentence:")
    if st.button("Predict Next Word"):
        result = generator(input_text, max_length=len(input_text.split()) + 5, num_return_sequences=1)
        st.write("Prediction:", result[0]['generated_text'])

elif task == "Story Generation":
    story_generator = load_text_generator()
    story_prompt = st.text_area("Provide a story prompt:")
    if st.button("Generate Story"):
        output = story_generator(
            story_prompt,
            max_length=250,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            num_return_sequences=1
        )
        st.write("Generated Story:", output[0]['generated_text'])

elif task == "Chatbot":
    tokenizer, model = load_dialo_model()

    if "chat_history_ids" not in st.session_state:
        st.session_state.chat_history_ids = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    user_input = st.text_input("You:", key="chat_input")

    if st.button("Chat") and user_input.strip() != "":
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Append new user input to chat history
        if st.session_state.chat_history_ids is not None:
            bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9
        )

        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Save full chat history
        st.session_state.chat_history_ids = chat_history_ids
        st.session_state.conversation.append(("You", user_input))
        st.session_state.conversation.append(("Bot", response))

    if st.button("Clear Chat"):
        st.session_state.chat_history_ids = None
        st.session_state.conversation = []

    # Display entire chat
    st.markdown("### Conversation History")
    for speaker, msg in st.session_state.conversation:
        st.markdown(f"**{speaker}:** {msg}")


elif task == "Sentiment Analysis":
    sentiment = load_sentiment()
    review = st.text_area("Enter your opinion:")
    if st.button("Analyze Sentiment"):
        result = sentiment(review)
        st.write("Sentiment:", result[0]['label'], "| Confidence:", result[0]['score'])

elif task == "Question Answering":
    qa = load_qa()
    context = st.text_area("Enter context:")
    question = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        answer = qa(question=question, context=context)
        st.write("Answer:", answer['answer'])

elif task == "Image Generation":
    pipe = load_image_generator()
    prompt = st.text_input("Enter a prompt for the image:")
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)
        #A futuristic cityscape at sunset
