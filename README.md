# LLM Chatbot for Messengers

## Description

Welcome to the LLM Chatbot for Messengers! This open-source project aims to simplify the creation of chatbots for various messaging platforms. With a focus on ease of use and extensibility, this project allows developers to quickly build and deploy chatbots that leverage advanced language models.

## Environment

- **Language:** Python 3.11 (currently supports only 3.11)
- **Project Manager:** Hatch (Python project manager)
- **Package Manager:** uv (Python fast package manager)

## Version

Current version: **0.1.0**

## Load Map

- **Current Feature:** Kakao Chatbot
- **Next Feature:** Adding Observability with Langfuse

## Getting Started

### Prerequisites

Make sure you have Python 3.11 installed. You can download it from the [official Python website](https://www.python.org/downloads/).

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/llm-chatbot-for-messengers.git
   cd llm-chatbot-for-messengers
   ```

2. Install dependencies using Hatch:

   ```bash
   hatch install
   ```

3. Start the chatbot:

   ```bash
   hatch run start
   ```
#### Note
- Execute `hatch config set dirs.env.virtual .venv` to create environments in the project.
### Usage

Once the chatbot is running, you can interact with it through the Kakao messaging platform. Follow the instructions provided in the documentation to set up your Kakao bot.

## Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, feel free to open an issue on the GitHub repository or contact the maintainers.

---

Feel free to modify any section to better fit your project's specifics!
