import gradio as gr
from huggingface_hub import InferenceClient
from transformers.agents import stream_to_gradio
from dataclasses import asdict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client with token from .env
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

client = InferenceClient(
    "HuggingFaceH4/zephyr-7b-beta",
    token=HUGGINGFACE_TOKEN
)

def process_url(url):
    """Process the URL and generate quiz questions"""
    return "Generating quiz from URL: " + url

def welcome_message():
    """Returns the welcome message displayed when the application starts."""
    return """
    Welcome to the Quiz Generator!
    This application helps you create quizzes based on content from URLs.
    
    """

def interact_with_agent(prompt, history):
    """Handles chat interaction and generates responses.

    Args:
        prompt (str): User's input message
        history (list): Chat history
    """
    messages = []
    
    # Add user message
    messages.append({"role": "user", "content": prompt})
    yield messages
    
    # Generate bot response
    response = client.text_generation(
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        stream=True
    )
    
    bot_message = {"role": "assistant", "content": ""}
    messages.append(bot_message)
    
    # Stream the response
    for token in response:
        bot_message["content"] += token
        yield messages
    
    yield messages

main_interface = gr.ChatInterface(
    interact_with_agent,
    title="Quiz Generator",
    textbox=gr.Textbox(placeholder="Your message", container=False, scale=7),
    description=welcome_message(),
    type="messages"
).launch()
