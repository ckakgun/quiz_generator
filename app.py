"""A web-based quiz generator that creates interactive questions from URL content using LlamaIndex and Gemma."""

import re
import sys
import os
import requests

import gradio as gr
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import TrafilaturaWebReader
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def initialize_model():
    """Initialize and return the LLM model."""
    try:
        return ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Error initializing the model: {error_msg}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have set OPENAI_API_KEY in your environment")
        print("2. Check your internet connection")
        sys.exit(1)

# Initialize LLM
llm = initialize_model()


class WelcomeAgent:
    """Agent responsible for user interaction and welcome messages."""

    @staticmethod
    def get_welcome_message() -> str:
        """Return the welcome message displayed to users when they first interact with the quiz assistant."""
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
    """Responsible for fetching and processing URL content using LlamaIndex."""
    def __init__(self):
        self.loader = TrafilaturaWebReader()
        
    def fetch_url_content(self, url: str) -> str:
        """Fetch and process content from URL using LlamaIndex."""
        try:
            documents = self.loader.load_data(urls=[url])
            if not documents:
                raise ValueError("No content could be extracted from the URL")
            
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine(llm=llm)
            summary = query_engine.query("Summarize the main points of this content in 3-4 sentences.")
            
            content = f"Summary:\n{str(summary)}\n\nDetailed Content:\n{documents[0].text}"
            return content[:4000]
            
        except Exception as e:
            return f"Error fetching URL: {str(e)}"

class QuizGeneratorAgent:
    """Responsible for generating quiz questions."""
    
    def __init__(self, language_model):
        """Initialize quiz generator with a language model."""
        self.llm = language_model
        self.prompt = PromptTemplate.from_template(
            """Based on the following content, generate 5 multiple choice questions with 4 options each. 
            Make questions engaging and test understanding of key concepts.
            Format each question exactly like this:
            Q1: [Question]
            Options:
            a) [Option 1]
            b) [Option 2]
            c) [Option 3]
            d) [Option 4]
            Correct: [a/b/c/d]
            
            Content: {content}
            """
        )
        self.chain = self.prompt | self.llm | self._format_response
    
    def _format_response(self, response) -> dict:
        """Format the LLM response into structured quiz data."""
        raw_text = response.content if hasattr(response, 'content') else str(response)
        return self.parse_quiz_content(raw_text)
    
    def generate_quiz(self, content: str) -> dict:
        """Generate quiz questions from content."""
        return self.chain.invoke({"content": content})
    
    @staticmethod
    def parse_quiz_content(quiz_text: str) -> dict:
        """Parse quiz text into structured format."""
        questions = []
        current_question = {}
        current_options = {}
        
        for line in quiz_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Q'):
                if current_question and current_options:
                    current_question['options'] = current_options
                    questions.append(current_question)
                current_question = {
                    'question': line.split(':', 1)[1].strip(),
                    'options': {},
                    'correct': None
                }
                current_options = {}
            elif line.startswith(('a)', 'b)', 'c)', 'd)')):
                option_letter = line[0]
                option_text = line[2:].strip()
                current_options[option_letter] = option_text
            elif line.startswith('Correct:'):
                current_question['correct'] = line.split(':')[1].strip()
        
        if current_question and current_options:
            current_question['options'] = current_options
            questions.append(current_question)
            
        return {
            'questions': questions,
            'total_questions': len(questions),
            'current_question': 0,
            'quiz_completed': False
        }

    def format_current_question(self, quiz_data: dict) -> str:
        """Format the current question for display."""
        if not quiz_data or quiz_data['quiz_completed']:
            return "Quiz completed!"
            
        current_idx = quiz_data['current_question']
        if current_idx >= len(quiz_data['questions']):
            return "Quiz completed!"
            
        question = quiz_data['questions'][current_idx]
        formatted = f"### Question {current_idx + 1} of {quiz_data['total_questions']}\n\n"
        formatted += f"**{question['question']}**\n\n"
        
        for option_letter, option_text in sorted(question['options'].items()):
            formatted += f"{option_letter}) {option_text}\n"
            
        formatted += "\nEnter your answer (a/b/c/d):"
        return formatted

class QuizSubmissionAgent:
    """Handles the submission and validation of quiz answers."""
    
    def __init__(self):
        self.submissions = {}
        
    def submit_answer(self, question_index: int, user_answer: str, correct_answer: str) -> dict:
        """
        Submit an answer for a specific question.
        Returns submission status and validation result.
        """
        if not self._validate_answer_format(user_answer):
            return {
                'status': 'error',
                'message': 'Please enter a valid answer (a, b, c, or d)'
            }
            
        self.submissions[question_index] = {
            'user_answer': user_answer.lower(),
            'correct_answer': correct_answer.lower(),
            'is_correct': user_answer.lower() == correct_answer.lower()
        }
        
        return {
            'status': 'success',
            'message': 'Answer submitted successfully',
            'submission': self.submissions[question_index]
        }
    
    def get_submission(self, question_index: int) -> dict:
        """Get the submission for a specific question."""
        return self.submissions.get(question_index)
    
    def get_all_submissions(self) -> dict:
        """Get all submissions for the quiz."""
        return self.submissions
    
    def clear_submissions(self):
        """Clear all submissions."""
        self.submissions = {}
    
    @staticmethod
    def _validate_answer_format(answer: str) -> bool:
        """Validate if the answer format is correct."""
        return bool(answer and answer.lower() in ['a', 'b', 'c', 'd'])

class ScoreCalculatorAgent:
    """Calculates and formats quiz scores."""
    
    @staticmethod
    def calculate_score(submissions: dict) -> dict:
        """
        Calculate the total score and generate detailed results.
        """
        if not submissions:
            return {
                'score': 0,
                'total_questions': 0,
                'percentage': 0,
                'summary': "No submissions found."
            }
        
        correct_count = sum(1 for sub in submissions.values() if sub['is_correct'])
        total_questions = len(submissions)
        percentage = (correct_count / total_questions) * 100
        
        # Generate detailed summary
        summary = ScoreCalculatorAgent._format_score_summary(
            correct_count, 
            total_questions, 
            percentage, 
            submissions
        )
        
        return {
            'score': correct_count,
            'total_questions': total_questions,
            'percentage': percentage,
            'summary': summary
        }
    
    @staticmethod
    def _format_score_summary(correct: int, total: int, percentage: float, submissions: dict) -> str:
        """Format the score summary with detailed feedback."""
        summary = [
            "### 📊 Quiz Results\n",
            f"## Final Score: {correct}/{total} ({percentage:.1f}%)\n\n"
        ]
        
        # Add performance indicator
        if percentage >= 80:
            summary.append("🌟 Excellent performance!\n")
        elif percentage >= 60:
            summary.append("👍 Good job!\n")
        else:
            summary.append("💪 Keep practicing!\n")
        
        # Detailed question breakdown
        summary.append("\n### Question Breakdown:\n")
        for q_idx, submission in sorted(submissions.items()):
            status = "✅" if submission['is_correct'] else "❌"
            summary.append(f"\n**Question {q_idx + 1}**: {status}\n")
            summary.append(f"Your answer: {submission['user_answer'].upper()}\n")
            if not submission['is_correct']:
                summary.append(f"Correct answer: {submission['correct_answer'].upper()}\n")
        
        return "".join(summary)

class QuizApp:
    """Main application class integrating all components."""

    def __init__(self):
        self.welcome_agent = WelcomeAgent()
        self.content_agent = ContentAgent()
        self.quiz_generator = QuizGeneratorAgent(llm)
        self.submission_agent = QuizSubmissionAgent()
        self.score_calculator = ScoreCalculatorAgent()
        self.current_quiz = None

    def generate_quiz(self, message: str):
        """Generate quiz questions from content."""
        if not message:
            return [{"role": "assistant", "content": "Please provide some content or a URL to generate quiz questions."}]
            
        try:
            if self.welcome_agent.validate_url(message):
                print("Fetching and processing URL content...")
                content = self.content_agent.fetch_url_content(message)
                if content.startswith("Error"):
                    return [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": content}
                    ]
            else:
                content = message
            
            print("Generating quiz questions...")
            self.current_quiz = self.quiz_generator.generate_quiz(content)
            self.submission_agent.clear_submissions()
            initial_question = self.quiz_generator.format_current_question(self.current_quiz)
            
            return [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Quiz generated! Let's begin:\n\n" + initial_question}
            ]
            
        except requests.RequestException as e:
            return [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"Error fetching URL: Network or server error - {str(e)}"}
            ]
        except ValueError as e:
            return [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"Error processing content: {str(e)}"}
            ]
        except (KeyError, IndexError) as e:
            return [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"Error parsing quiz content: {str(e)}"}
            ]
        except TimeoutError as e:
            return [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "The request timed out. Please try again or try with a different URL."}
            ]

    def submit_answer(self, answer: str, history):
        """Handle user's answer submission."""
        if not self.current_quiz or self.current_quiz['quiz_completed']:
            return history

        current_idx = self.current_quiz['current_question']
        current_question = self.current_quiz['questions'][current_idx]
        
        # Submit answer through submission agent
        submission_result = self.submission_agent.submit_answer(
            current_idx,
            answer,
            current_question['correct']
        )
        
        if submission_result['status'] == 'error':
            history.extend([
                {"role": "user", "content": answer},
                {"role": "assistant", "content": submission_result['message']}
            ])
            return history
        
        # Move to next question
        self.current_quiz['current_question'] += 1
        
        # Check if quiz is completed
        if self.current_quiz['current_question'] >= self.current_quiz['total_questions']:
            self.current_quiz['quiz_completed'] = True
            # Calculate final score
            score_results = self.score_calculator.calculate_score(
                self.submission_agent.get_all_submissions()
            )
            history.extend([
                {"role": "user", "content": answer},
                {"role": "assistant", "content": score_results['summary']}
            ])
            return history
        
        # Show next question
        next_question = self.quiz_generator.format_current_question(self.current_quiz)
        history.extend([
            {"role": "user", "content": answer},
            {"role": "assistant", "content": next_question}
        ])
        return history

    def create_interface(self):
        """Create and launch the Gradio chat interface for the quiz application."""
        with gr.Blocks(title="Quiz Generator") as interface:
            gr.Markdown("# Quiz Generator")
            gr.Markdown("Share a URL or paste content to generate interactive quiz questions.")
            
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": self.welcome_agent.get_welcome_message()}],
                height=450,
                show_label=False,
                container=True,
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Paste a URL here to generate quiz questions, or enter a/b/c/d to answer",
                    container=False,
                    scale=7
                )
                submit_btn = gr.Button("Submit", scale=1)
            
            
            def handle_submit(message: str, history):
                """Handle both quiz generation and answer submission."""
                history = history or []
                if not self.current_quiz or message.startswith('http'):
                    return self.generate_quiz(message), ""
                return self.submit_answer(message, history), ""
            
            
            
            submit_btn.click(
                fn=handle_submit,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                fn=handle_submit,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            gr.Examples(
                examples=[
                    "https://en.wikipedia.org/wiki/Artificial_intelligence",
                    "https://en.wikipedia.org/wiki/Machine_learning"
                ],
                inputs=msg,
                label="Example URLs"
            )

        interface.launch(
            share=False,
            show_error=True,
            server_name="127.0.0.1",
            server_port=7860,
            quiet=False
        )

if __name__ == "__main__":
    app = QuizApp()
    app.create_interface()
