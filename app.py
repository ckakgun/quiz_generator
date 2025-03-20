"""A web-based quiz generator that creates interactive questions from URL content using LlamaIndex and OpenAI."""

import re 
import os

import gradio as gr
import requests
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.web import SimpleWebPageReader
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

class WelcomeAgent:
    """Agent responsible for user interaction and welcome messages."""

    @staticmethod
    def get_welcome_message() -> str:
        """Return the welcome message displayed to users when they first interact with the quiz assistant."""
        return """👋 Hi! I'm your quiz assistant. I can create interactive quizzes from any web content.

        First, please enter your OpenAI API key to start using the quiz generator."""

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
        self.loader = SimpleWebPageReader()
        
    def fetch_url_content(self, url: str) -> str:
        """Fetch and process content from URL using LlamaIndex."""
        try:
            documents = self.loader.load_data(urls=[url])
            if not documents:
                raise ValueError("No content could be extracted from the URL")
            
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()
            summary = query_engine.query("Summarize the main points of this content in 3-4 sentences.")
            
            content = f"Summary:\n{str(summary)}\n\nDetailed Content:\n{documents[0].text}"
            return content[:4000]
            
        except requests.RequestException as e:
            return f"Error fetching URL: {str(e)}"

class QuizGeneratorAgent:
    """Responsible for generating quiz questions."""
    
    def __init__(self, language_model):
        """Initialize quiz generator with a language model."""
        self.llm = language_model
        self.prompt_template = """Based on the following content, generate 5 multiple choice questions with 4 options each. 
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
    
    def generate_quiz(self, content: str) -> dict:
        """Generate quiz questions from content."""
        prompt = self.prompt_template.format(content=content)
        response = self.llm.invoke(prompt)
        return self.parse_quiz_content(response.content)
    
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
        self.content_agent = None
        self.quiz_generator = None
        self.submission_agent = None
        self.score_calculator = None
        self.current_quiz = None
        self.api_key_validated = False

    def validate_api_key(self, api_key: str) -> tuple[bool, str]:
        """Validate OpenAI API key."""
        if not api_key or len(api_key.strip()) < 20:  # Basic length check
            return False, "Please enter a valid OpenAI API key"
            
        try:
            os.environ["OPENAI_API_KEY"] = api_key.strip()
            # Initialize components only after API key is validated
            Settings.embed_model = OpenAIEmbedding()
            llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                temperature=0.7,
                max_tokens=512,
                top_p=0.9,
            )
            
            # Initialize components
            self.content_agent = ContentAgent()
            self.quiz_generator = QuizGeneratorAgent(llm)
            self.submission_agent = QuizSubmissionAgent()
            self.score_calculator = ScoreCalculatorAgent()
            self.api_key_validated = True
            
            return True, "API key validated successfully! You can now start generating quizzes."
        except Exception as e:
            return False, f"Invalid API key or error occurred: {str(e)}"

    def generate_quiz(self, message: str):
        """Generate quiz questions from content."""
        if not self.api_key_validated:
            return [], "Please enter your OpenAI API key first to start using the quiz generator."
            
        if not message:
            return [], "Please provide some content or a URL to generate quiz questions."
            
        try:
            if self.welcome_agent.validate_url(message):
                print("Fetching and processing URL content...")
                content = self.content_agent.fetch_url_content(message)
                if content.startswith("Error"):
                    return [], content
            else:
                content = message
            
            print("Generating quiz questions...")
            self.current_quiz = self.quiz_generator.generate_quiz(content)
            self.submission_agent.clear_submissions()
            initial_question = self.quiz_generator.format_current_question(self.current_quiz)
            
            return [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Quiz generated! Let's begin:\n\n" + initial_question}
            ], ""
            
        except Exception as e:
            return [], f"Error: {str(e)}"

    def submit_answer(self, answer: str, history):
        """Handle user's answer submission."""
        if not self.current_quiz or self.current_quiz['quiz_completed']:
            return history, gr.update(value="", interactive=True)

        current_idx = self.current_quiz['current_question']
        current_question = self.current_quiz['questions'][current_idx]
        
        # Submit answer through submission agent
        submission_result = self.submission_agent.submit_answer(
            current_idx,
            answer,
            current_question['correct']
        )
        
        if submission_result['status'] == 'error':
            history.append({"role": "user", "content": answer})
            history.append({"role": "assistant", "content": submission_result['message']})
            return history, gr.update(value="", interactive=True)
        
        # Move to next question
        self.current_quiz['current_question'] += 1
        
        # Check if quiz is completed
        if self.current_quiz['current_question'] >= self.current_quiz['total_questions']:
            self.current_quiz['quiz_completed'] = True
            # Calculate final score
            score_results = self.score_calculator.calculate_score(
                self.submission_agent.get_all_submissions()
            )
            history.append({"role": "user", "content": answer})
            history.append({"role": "assistant", "content": score_results['summary']})
            return history, gr.update(value="", interactive=True)
        
        # Show next question
        next_question = self.quiz_generator.format_current_question(self.current_quiz)
        history.append({"role": "user", "content": answer})
        history.append({"role": "assistant", "content": next_question})
        return history, gr.update(value="", interactive=True)

    def create_interface(self):
        """Create and launch the Gradio chat interface for the quiz application."""
        with gr.Blocks(title="Quiz Generator") as interface:
            gr.Markdown("# Quiz Generator")
            gr.Markdown("First, enter your OpenAI API key to start.")
            
            with gr.Row():
                api_key_input = gr.Textbox(
                    placeholder="Enter your OpenAI API key here",
                    type="password",
                    label="OpenAI API Key",
                    scale=4
                )
                validate_btn = gr.Button("Validate Key", scale=1)
            
            api_status = gr.Markdown("⚠️ Please enter your OpenAI API key to start")
            
            chatbot = gr.Chatbot(
                value=[],
                height=450,
                show_label=False,
                container=True,
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="After validating API key, paste a URL here to generate quiz questions, or enter a/b/c/d to answer",
                    container=False,
                    scale=7,
                    interactive=False
                )
                submit_btn = gr.Button("Submit", scale=1, interactive=False)
            
            def validate_key(api_key):
                """Validate OpenAI API key and update UI accordingly."""
                try:
                    is_valid, message = self.validate_api_key(api_key)
                    if is_valid:
                        return {
                            api_status: gr.Markdown("✅ " + message),
                            msg: gr.update(interactive=True),
                            submit_btn: gr.update(interactive=True),
                            chatbot: [{"role": "assistant", "content": self.welcome_agent.get_welcome_message()}]
                        }
                    return {
                        api_status: gr.Markdown("❌ " + message),
                        msg: gr.update(interactive=False),
                        submit_btn: gr.update(interactive=False),
                        chatbot: []
                    }
                except Exception as e:
                    return {
                        api_status: gr.Markdown("❌ Error validating key: " + str(e)),
                        msg: gr.update(interactive=False),
                        submit_btn: gr.update(interactive=False),
                        chatbot: []
                    }
            
            def handle_submit(message, history):
                """Handle both quiz generation and answer submission."""
                if not self.api_key_validated:
                    return history, ""
                    
                history = history or []
                if not self.current_quiz or message.startswith('http'):
                    new_history, error = self.generate_quiz(message)
                    if error:
                        history.append({"role": "user", "content": message})
                        history.append({"role": "assistant", "content": error})
                        return history, gr.update(value="", interactive=True)
                    return new_history, gr.update(value="", interactive=True)
                
                history, _ = self.submit_answer(message, history)
                return history, gr.update(value="", interactive=True)
            
            # Event handlers
            validate_btn.click(
                fn=validate_key,
                inputs=[api_key_input],
                outputs=[api_status, msg, submit_btn, chatbot]
            )
            
            submit_btn.click(
                fn=handle_submit,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            # Also handle Enter key in message box
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
            server_name="0.0.0.0",
            share=True,
            show_error=True
        )

if __name__ == "__main__":
    app = QuizApp()
    app.create_interface()
