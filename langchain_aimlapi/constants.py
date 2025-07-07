# AIMLAPI_HEADERS: sent with each request to let the server know who you are
# Helps with analytics, debugging, and enforcing usage policies
AIMLAPI_HEADERS = {
    # Tells the API which application is making the call
    "HTTP-Referer": "https://github.com/langchain-ai/langchain",
    # Identifies the client or library name for tracking
    "X-Title": "LangChain",
}
