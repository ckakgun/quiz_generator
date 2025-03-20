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

## Installation

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

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser and go to `http://localhost:7862`

3. Enter your OpenAI API key when prompted

4. Paste any URL or content to generate a quiz

5. Answer the questions and get instant feedback

## Dependencies

Core dependencies:
- `gradio`: Web interface framework
- `openai`: OpenAI API client
- `langchain-openai`: LangChain OpenAI integration
- `llama-index-core`: Core LlamaIndex functionality
- `llama-index-readers-web`: Web content extraction
- `llama-index-embeddings-openai`: OpenAI embeddings support
- `python-dotenv`: Environment variable management
- `requests`: HTTP client

## Troubleshooting

Common issues and solutions:

1. **OpenAI API Key Issues**:
   - Make sure you've entered a valid OpenAI API key
   - Check if your API key has sufficient credits
   - Ensure you have billing set up in your OpenAI account

2. **Web Content Extraction Issues**:
   - Some websites may block content extraction
   - Try with different URLs if you encounter issues
   - Make sure the URL is accessible

## License

This project is licensed under the MIT License 

## Support

If you encounter any issues:
1. Check the Troubleshooting section above
2. Open an issue on GitHub
3. Make sure your dependencies are up to date

---
*Made with ❤️ for learning*
