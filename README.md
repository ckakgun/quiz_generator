---
title: Quiz Generator
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.21.0
app_file: app.py
pinned: false
---

# Quiz Generator

![Quiz Generator Interface](assets/demo.png)

An intelligent web application that automatically generates interactive quizzes from any web content using LlamaIndex and OpenAI. Perfect for educators, students, and lifelong learners who want to create engaging quizzes from online resources.

## Features

- 🌐 **URL-based Quiz Generation**: Simply paste any URL to generate questions
- 🤖 **AI-Powered**: Utilizes OpenAI's GPT models for intelligent question generation
- ✨ **Interactive Interface**: Clean, user-friendly Gradio interface
- 📝 **Multiple Choice Format**: Automatically generates 4 options per question
- 📊 **Real-time Scoring**: Instant feedback and score tracking
- 🔄 **Continuous Learning**: Learn from any web content interactively

## Prerequisites

Before you begin, ensure you have:
- Python 3.9 or higher installed
- An OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git installed (for GitHub installation)

## Installation Options

### Option 1: Running Locally via GitHub

1. Clone the repository:
```bash
git clone https://github.com/ckakgun/quiz-generator
cd quiz-generator
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file in the project root:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:7860`

### Option 2: Using Hugging Face Spaces

1. Visit the Space: [Quiz Generator on Hugging Face](https://huggingface.co/spaces/24-cka-ML/quiz-generator)

2. To create your own instance:
   - Click "Duplicate Space" on the top right
   - In your duplicated Space's settings:
     1. Go to "Repository Secrets"
     2. Add a new secret:
        - Name: `OPENAI_API_KEY`
        - Value: Your OpenAI API key
   - The Space will automatically rebuild and deploy

## Usage Guide

1. Access the application through your browser
2. Paste any URL containing the content you want to create a quiz from
3. Click "Submit" or press Enter
4. Answer the generated multiple-choice questions
5. Get instant feedback and see your final score

## Project Structure

```
quiz-generator/
├── app.py           # Main application file
├── requirements.txt # Project dependencies
├── .env            # Environment variables (not tracked in git)
└── README.md       # Project documentation
```

## Dependencies

Key dependencies include:
- gradio>=4.0.0
- llama-index-core
- llama-index-readers-web
- llama-index-llms-openai
- llama-index-embeddings-openai
- langchain-core
- langchain-openai
- python-dotenv
- openai
- requests

## Troubleshooting

Common issues and solutions:

1. **ModuleNotFoundError**:
   - Ensure you've installed all dependencies: `pip install -r requirements.txt`
   - Make sure you're in the virtual environment

2. **API Key Error**:
   - Check if your OpenAI API key is correctly set in .env file (local) or Space secrets (Hugging Face)
   - Verify the API key is valid and has sufficient credits

3. **Web Content Extraction Issues**:
   - Some websites may block content extraction
   - Try with different URLs if you encounter issues
   - Make sure the URL is accessible and contains readable content

4. **Gradio Interface Issues**:
   - Clear your browser cache
   - Try a different browser
   - Check if the port 7860 is available

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License 

## Support

If you encounter any issues or have questions:
1. Check the Troubleshooting section above
2. Open an issue on GitHub
3. Ask on the Hugging Face Space discussion tab

---
*Made with ❤️ for learning*
