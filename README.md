---
title: Quiz Generator
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
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

## Tech Stack

- **Frontend**: Gradio
- **Backend**: Python
- **NLP Processing**: LlamaIndex
- **AI Model**: OpenAI GPT
- **Web Scraping**: Trafilatura

## Installation

1. Clone the repository:
```bash
git clone https://huggingface.co/spaces/[your-username]/quiz-generator
cd quiz-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file and add your OpenAI API key
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:7860`

## Usage

1. Open the application in your web browser
2. Paste any URL containing the content you want to create a quiz from
3. Click "Submit" or press Enter
4. Answer the generated multiple-choice questions
5. Get instant feedback and final score

## Project Structure

```
quiz-generator/
├── app.py           # Main application file
├── requirements.txt # Project dependencies
├── .env            # Environment variables (not tracked in git)
├── README.md       # Project documentation
└── assets/         # Images and other assets
    └── demo.png    # Application screenshot
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| OPENAI_API_KEY | Your OpenAI API key | Yes |

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for document processing
- [OpenAI](https://openai.com/) for the language model
- [Gradio](https://gradio.app/) for the web interface



---
*Made with ❤️ for learning*
