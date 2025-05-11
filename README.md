# Retell AI Call Analysis

A Python application for analyzing and monitoring Retell AI phone calls, providing automated analysis of call outcomes, issue detection, and notification capabilities.

## Overview

This project provides a comprehensive solution for analyzing Retell AI phone calls. It:

1. Retrieves call data from the Retell API
2. Analyzes call transcripts using OpenAI's GPT models
3. Stores analysis results in a SQLite database
4. Sends notifications about calls requiring human review via Telegram

The system is designed to handle large volumes of calls efficiently and provide actionable insights about call performance.

## Features

- **Automated Call Analysis**: Processes call transcripts to determine success status, issue detection, and key metrics
- **Database Storage**: Stores all call analysis results in a SQLite database for historical tracking
- **Telegram Notifications**: Sends alerts for calls that require human review
- **Scheduled Operation**: Includes a shell script for running the analysis during business hours
- **Configurable Analysis**: Customizable analysis criteria for different business needs

## Installation

### Prerequisites

- Python 3.12+
- Retell API key
- OpenAI API key
- Telegram Bot token (optional, for notifications)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/retell-ai-call-analysis.git
   cd retell-ai-call-analysis
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Create a `.env` file with your API keys:
   ```
   RETELL_API_KEY=your_retell_api_key
   OPENAI_API_KEY=your_openai_api_key
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token  # Optional
   TELEGRAM_CHAT_ID=your_telegram_chat_id      # Optional
   ```

## Usage

### Setting Up the Database

Initialize the database with:
```bash
python -m retell_ai_call_analysis.db.setup_database --db-path data/call_analysis.db
```

### Running the Analysis

Run the analysis script:

```bash
python -m retell_ai_call_analysis.run
```

### Scheduled Operation

The included `run.sh` script can be used with cron to schedule regular analysis during business hours:

```bash
chmod +x run.sh
# Add to crontab to run every hour during business hours
# 0 8-20 * * * /path/to/retell-ai-call-analysis/run.sh
```

## Architecture

The project is structured as follows:

- `retell_ai_call_analysis/`: Main package
  - `clients/`: API client implementations
  - `db/`: Database models and setup
  - `model/`: Data models
  - `run.py`: Main entry point

## Configuration

The analysis can be configured by modifying the analysis prompt in `db_call_analysis_client.py`. This allows for customization of:

- Success criteria
- Issue detection parameters
- Human review thresholds

## Development

### Adding New Features

1. Create a new branch for your feature
2. Implement and test your changes
3. Submit a pull request

### Running Tests

```bash
pytest
```

## License

[MIT License](LICENSE)

## Acknowledgements

- [Retell AI](https://retellai.com) for their API
- [OpenAI](https://openai.com) for their language models
