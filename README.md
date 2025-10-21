# STOP THE SLOP

<img width="720" height="746" alt="sts" src="https://github.com/user-attachments/assets/4eed5b93-a666-45ff-ac6d-0595160529f8" />


## Install

```bash
bash installer.sh
cp .env.example .env  # optional for CLI
source .venv-cyberslop-py3*/bin/activate
```

## Run

```bash
streamlit run app_streamlit.py
# or
python cli.py --help
```

## Mermaid workflow

```mermaid
flowchart TD
A[Input URL or File or Text] --> B[Ingestors: newspaper3k and optional Selenium and file parsers]
B --> C[Text normalization]
C --> D[Rules Engine JSON rules]
D --> E[CTI Audit Persona Prompt]
E --> F[Provider Adapters: OpenAI Anthropic Gemini]
F --> G[Structured JSON verdicts]
D --> H[Scoring Aggregator]
G --> H
H --> I[Decision and reasons]
I --> K[Friction Generator]
K --> J[Report JSON and Markdown]
```

## Secrets configuration

For Streamlit runs, put secrets in `.streamlit/secrets.toml`.

```toml
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "..."
GOOGLE_API_KEY = "..."
SLOPWATCH_MAX_CHARS = 200000
SLOPWATCH_TIMEOUT_SEC = 20
```

The code reads secrets in this order: st.secrets then `.streamlit/secrets.toml` then environment variables.
The CLI still supports `.env` via python-dotenv. You can also place a `.streamlit/secrets.toml` in the project and the CLI will read it.

## Threat model

SSRF guard, timeouts, JSON only outputs, and no code execution.

## License

Apache 2.0
