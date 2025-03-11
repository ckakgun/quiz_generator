import os
import re
from dataclasses import asdict

import gradio as gr
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Initialize client with token from .env
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Using Mixtral model directly
    token=HUGGINGFACE_TOKEN
)

def fetch_url_content(url):
    """Fetch and parse content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text[:4000]

    except Exception as e:
        return f"Error fetching URL: {str(e)}"

def format_quiz_markdown(quiz_text):
    """Format quiz into clickable markdown with hidden answers."""
    questions = []
    answers = []
    current_item = ""
    
    for line in quiz_text.split('\n'):
        line = line.strip()
        if line.startswith('Q'):
            if current_item:
                questions.append(current_item)
            current_item = line
        elif line.startswith('A'):
            answers.append(line)
            
    if current_item:
        questions.append(current_item)
    
    formatted_quiz = "### 📝 Generated Quiz\n\n"
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        # Remove Q1:, A1: prefixes
        q = q.split(':', 1)[1].strip()
        a = a.split(':', 1)[1].strip()
        
        formatted_quiz += f"**Question {i}**\n"
        formatted_quiz += f"{q}\n\n"
        formatted_quiz += f"<details><summary>Click to see answer</summary>\n\n"
        formatted_quiz += f"*{a}*\n\n"
        formatted_quiz += "</details>\n\n"
    
    return formatted_quiz

def generate_quiz(content):
    """Generates quiz questions from content."""
    prompt = f"""Based on the following content, generate 5 quiz questions with their answers. 
    Make questions engaging and diverse (multiple choice, true/false, open-ended).
    Format each question with its answer exactly like this:
    Q1: [Question]
    A1: [Answer]

    Content: {content}
    """
    response = client.text_generation(
        prompt,
        max_new_tokens=1000,
        temperature=0.7,
        do_sample=True,
        return_full_text=False
    )
    return format_quiz_markdown(response)

def is_valid_url(text):
    """Check if the text contains a URL."""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return bool(url_pattern.search(text))

def welcome_message():
    """Returns the welcome message displayed when the application starts."""
    return """
    # 🎯 Welcome to the Quiz Generator!
    
    This application helps you create interactive quizzes based on content from URLs.
    
    ### How to use:
    1. 🔗 Paste a URL in the textbox below
    2. 🤖 AI will generate quiz questions
    3. 👆 Click on answers to reveal them
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
    
    # Check if input contains URL
    if not is_valid_url(prompt):
        bot_message = {"role": "assistant", "content": "⚠️ Please provide a valid URL to generate quiz questions."}
        messages.append(bot_message)
        yield messages
        return
    
    # Fetch content from URL
    content = fetch_url_content(prompt)
    if content.startswith("Error"):
        bot_message = {"role": "assistant", "content": f"❌ {content}"}
        messages.append(bot_message)
        yield messages
        return
    
    # Generate quiz
    quiz = generate_quiz(content)
    
    bot_message = {"role": "assistant", "content": quiz}
    messages.append(bot_message)
    yield messages

# Configure the interface
main_interface = gr.ChatInterface(
    interact_with_agent,
    title="🎓 Interactive Quiz Generator",
    textbox=gr.Textbox(
        placeholder="Paste a URL here to generate quiz questions",
        container=False,
        scale=7
    ),
    description=welcome_message(),
    type="messages"
).launch()
