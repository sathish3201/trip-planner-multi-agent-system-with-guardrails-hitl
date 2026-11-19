
import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

TOKEN_LIMIT = int(os.getenv("TOKEN_LIMIT", "7500"))
LOCAL_TPM_LIMIT = int(os.getenv("LOCAL_TPM_LIMIT", "8000"))

MAX_USER_INPUT_CHARS = 7500
MAX_OUTPUT_TOKENS = 1200

TOKEN_ESTIMATE_CHARS_PER_TOKEN = 2.5

MAX_429_RETRIES = 4
RETRY_BASE_SECONDS = 2
RETRY_MAX_SECONDS = 30

MAX_CONCURRENT_REQUESTS = 2

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing.")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing.")
