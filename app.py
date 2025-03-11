import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def process_url(url):
    """Process the URL and generate quiz questions"""
    return "Generating quiz from URL: " + url

def welcome_message():
    """Returns the welcome message displayed when the application starts."""
    return """
    Welcome to the Quiz Generator!
    This application helps you create quizzes based on content from URLs.
    
    """

main_interface = gr.ChatInterface(
    fn=process_url,
    title="Quiz Generator",
    textbox=gr.Textbox(placeholder="Your message", container=False, scale=7),
    description=welcome_message(),
    type="messages"
).launch()
