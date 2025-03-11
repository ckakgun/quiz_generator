import os
import re
from typing import List

import gradio as gr
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize Hugging Face client
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

# Initialize LLM
llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    model_kwargs={"temperature": 0.7}
)

class ContentExtractor:
    """Responsible for fetching and processing URL content."""
    
    @staticmethod
    def fetch_url_content(url: str) -> str:
        """Fetch and clean content from URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            # Get main content
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:4000]  # Limit content length
            
        except Exception as e:
            return f"Error fetching URL: {str(e)}"

class QuizGenerator:
    """Responsible for generating quiz questions."""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["content"],
            template="""Based on the following content, generate 5 quiz questions with their answers. 
            Make questions engaging and diverse (multiple choice, true/false, open-ended).
            Format each question with its answer exactly like this:
            Q1: [Question]
            A1: [Answer]
            
            Content: {content}
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def generate_quiz(self, content: str) -> str:
        """Generate quiz questions from content."""
        response = self.chain.run(content=content)
        return self.format_quiz_markdown(response)
    
    @staticmethod
    def format_quiz_markdown(quiz_text: str) -> str:
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
            q = q.split(':', 1)[1].strip() if ':' in q else q
            a = a.split(':', 1)[1].strip() if ':' in a else a
            
            formatted_quiz += f"**Question {i}**\n"
            formatted_quiz += f"{q}\n\n"
            formatted_quiz += f"<details><summary>Click to see answer</summary>\n\n"
            formatted_quiz += f"*{a}*\n\n"
            formatted_quiz += "</details>\n\n"
        
        return formatted_quiz

# Initialize components
content_extractor = ContentExtractor()
quiz_generator = QuizGenerator(llm)

def is_valid_url(text: str) -> bool:
    """Check if the text contains a URL."""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return bool(url_pattern.search(text))

def welcome_message() -> str:
    """Returns the welcome message displayed when the application starts."""
    return """
    # 🎯 Welcome to the Quiz Generator!
    
    This application uses LLAMA 2 to create interactive quizzes from web content.
    
    ### How it works:
    1. 🤖 Content Extractor fetches and processes the webpage
    2. 🧠 Quiz Generator creates engaging questions using LLAMA 2
    3. 📝 The system formats a beautiful interactive quiz
    
    ### How to use:
    1. 🔗 Paste a URL in the textbox below
    2. ⏳ Wait for processing
    3. 👆 Click on answers to reveal them
    """

def interact_with_system(prompt: str, history: List[dict]) -> dict:
    """Handles chat interaction using LangChain components.

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
    
    # Extract content from URL
    content = content_extractor.fetch_url_content(prompt)
    if content.startswith("Error"):
        bot_message = {"role": "assistant", "content": f"❌ {content}"}
        messages.append(bot_message)
        yield messages
        return
    
    # Generate quiz
    quiz = quiz_generator.generate_quiz(content)
    
    bot_message = {"role": "assistant", "content": quiz}
    messages.append(bot_message)
    yield messages

# Configure the interface
main_interface = gr.ChatInterface(
    interact_with_system,
    title="🎓 LLAMA 2 Quiz Generator",
    textbox=gr.Textbox(
        placeholder="Paste a URL here to generate quiz questions",
        container=False,
        scale=7
    ),
    description=welcome_message(),
    type="messages"
).launch()
