import os
import re
from typing import List, Dict

import gradio as gr
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize Hugging Face client
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

# Initialize LLM
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/Qwen/Qwen2.5-1.5B-Instruct",
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    task="text-generation",
    temperature=0.7,
    top_p=0.9,
    model_kwargs={
        "max_length": 2048,
    }
)

class WelcomeAgent:
    
    """Agent responsible for user interaction and welcome messages."""

    @staticmethod
    def get_welcome_message() -> str:
        return """👋 Hi! I'm your quiz assistant. I can create interactive quizzes from any web content.

Just share a URL with me, and I'll generate multiple-choice questions to help you learn!"""

    @staticmethod
    def validate_url(text: str) -> bool:
        """Check if the text contains a valid URL."""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return bool(url_pattern.search(text))

class ContentAgent:
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
        

class QuizGeneratorAgent:
    """Responsible for generating quiz questions."""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["content"],
            template="""Based on the following content, generate 5 quiz questions with their answers. 
            Make questions engaging and diverse (multiple choice, true/false).
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
    
class ScoringAgent:
    """Agent responsible for scoring quiz answers."""

    @staticmethod
    def score_quiz(user_answers: List[str], questions: List[Dict]) -> Dict:
        """Score user's quiz answers."""
        pass

class QuizApp:
    """Agent responsible for quiz generation."""

    def __init__(self):
        self.welcome_agent = WelcomeAgent()
        self.content_agent = ContentAgent()
        self.quiz_generator = QuizGeneratorAgent(llm)
        self.scoring_agent = ScoringAgent()

    def generate_quiz(self, content: str) -> str:
        """Generate quiz questions from content."""
        return self.quiz_generator.generate_quiz(content)

    def create_interface(self):
        interface = gr.ChatInterface(
            self.generate_quiz,
            title="Quiz Generator",
            textbox=gr.Textbox(
                placeholder="Paste a URL here to generate quiz questions",
                container=False,
                scale=7
            ),
            chatbot=gr.Chatbot(
                value=[[None,self.welcome_agent.get_welcome_message()]],
                elem_id="chatbot",
                height=450,
                show_label=False,
                container=True
            ),
            type="messages"
        )
        interface.launch()

    def score_quiz(self, user_answers: List[str], questions: List[Dict]) -> Dict:
        """Score user's quiz answers."""
        return self.scoring_agent.score_quiz(user_answers, questions)
    
app = QuizApp()
app.create_interface()
