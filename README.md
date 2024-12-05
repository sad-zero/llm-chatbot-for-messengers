# LLM Chatbot for Messengers

## Description

Welcome to the LLM Chatbot for Messengers! This open-source project simplifies the creation of chatbots for various messaging platforms. With a focus on ease of use and extensibility, developers can quickly build and deploy chatbots that leverage advanced language models.

## Key Features

- **Current:** Kakao Chatbot integration.
- **Next:** Extensible chatbot framework for other platforms.

## Roadmap

- **Current Release (v0.1.0):** Initial Kakao chatbot functionality.
- **Planned Features:** Support for multiple messaging platforms, enhanced language understanding, and customizable workflows.

## Environment

- **Language:** Python 3.11 (supports only this version for now)
- **Project Manager:** Hatch (Python project manager)
- **Package Manager:** uv (fast Python package manager)

## Getting Started

### Prerequisites

1. Install Python 3.11 from the [official Python website](https://www.python.org/downloads/).
2. Install Hatch by following the [Hatch documentation](https://hatch.pypa.io/latest/).

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/llm-chatbot-for-messengers.git
   cd llm-chatbot-for-messengers
   ```

2. Set up the environment with Hatch:

   ```bash
   hatch env create dev
   hatch config set dirs.env.virtual .venv
   ```

3. Start the chatbot:

   ```bash
   hatch run dev:kakao-api & tail -f dev-kakao.log
   ```

### Optional Setup

For persistent chat history:

1. Navigate to the database scripts folder:

   ```bash
   cd scripts/database
   ```

2. Start the database using Docker:

   ```bash
   docker compose up -d
   ```

3. Connect to the database and execute the provided `*.sql` files to initialize the schema.

### Usage

Once the chatbot is running, you can interact with it on the Kakao messaging platform. Detailed setup instructions for Kakao bot integration are available in the documentation.

## Quick Start

Here’s how to get started in minutes:

1. Clone the repository and install dependencies.
2. Run the chatbot with a single command:
   ```bash
   hatch run dev:kakao-api
   ```
3. Start chatting via Kakao.

## Contributing

We welcome contributions! Here’s how to get started:

1. Read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
2. Fork the repository and make your changes.
3. Submit a pull request with a detailed description.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Have questions or feedback? Open an issue on the GitHub repository or contact the maintainers.
