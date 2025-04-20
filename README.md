# LearnFlow

LearnFlow is an automated learning roadmap generator that creates structured, comprehensive learning paths for any topic. It leverages the Gemini AI API to generate detailed tutorials, create visual diagrams, and organize educational content in a clear, hierarchical structure.

## 🌟 Features

- **Automated Roadmap Generation**: Creates structured learning paths with major topics and subtopics
- **Detailed Tutorials**: Generates comprehensive markdown tutorials for each topic and subtopic
- **Visual Diagrams**: Creates Mermaid diagrams to visualize learning paths and concept relationships
- **Hierarchical Organization**: Content is organized with clear numerical prefixes (1.0, 1.1, etc.)
- **GitHub Integration**: Automatically creates repositories and manages content through pull requests
- **Smart Content Validation**: Ensures generated content meets quality and formatting standards

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Google Cloud account with Gemini API access
- GitHub account with personal access token

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LearnFlow.git
cd LearnFlow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with the following:
GEMINI_API_KEY=your_gemini_api_key
GITHUB_TOKEN=your_github_token
```

## 📖 Usage

1. Import and initialize LearnFlow:
```python
from src.learnflow import LearnFlow

learnflow = LearnFlow()
```

2. Generate a roadmap for any topic:
```python
# Generate a roadmap for "Machine Learning"
learnflow.generate_roadmap("Machine Learning")
```

3. The generated content will be:
   - Organized in a hierarchical folder structure
   - Pushed to a GitHub repository
   - Available as markdown files and Mermaid diagrams

## 📁 Project Structure

```
LearnFlow/
├── src/
│   ├── learnflow.py           # Main class and orchestration
│   ├── roadmap_generator.py   # Roadmap generation logic
│   ├── content_creator.py     # Tutorial and content creation
│   ├── file_organizer.py      # File management and organization
│   ├── github_manager.py      # GitHub repository management
│   └── prompts/              
│       └── content_prompts.yaml # AI prompts configuration
├── requirements.txt
└── README.md
```

## 🔧 Configuration

The system can be configured through:
- Environment variables in `.env`
- Prompt templates in `src/prompts/content_prompts.yaml`
- Content validation rules in the code

## 📝 Content Structure

Generated content follows a clear hierarchical structure:
- Main topic (1.0)
- Major topics (2.0, 3.0, etc.)
- Subtopics (2.1, 2.2, etc.)

Each tutorial includes:
- Introduction
- Core Concepts
- Practical Implementation
- Advanced Topics
- Conclusion and Next Steps

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini API for content generation
- GitHub API for repository management
- Mermaid for diagram generation

## ⚠️ Note

This is an automated content generation tool. While it strives for accuracy and quality, please review and validate the generated content for your specific needs. 
