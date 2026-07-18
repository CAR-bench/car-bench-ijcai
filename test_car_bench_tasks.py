import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load .env from current directory explicitly
env_path = Path(__file__).parent / ".env"
print(f"Looking for .env at: {env_path}")
load_dotenv(dotenv_path=env_path)

print("=" * 80)
print("?? CAR-BENCH TASK TEST CASES")
print("=" * 80)

# Check if key is loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("? GROQ_API_KEY not found!")
    print(f"Checked location: {env_path}")
    print("\nTrying environment variables directly...")
    # Try to read .env file manually
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith("GROQ_API_KEY"):
                    print(f"Found in file: {line[:40]}...")
    except:
        print("Could not read .env file")
    exit(1)
else:
    print(f"? GROQ_API_KEY loaded: {api_key[:20]}...\n")

# Initialize Groq
try:
    client = Groq(api_key=api_key)
    print("? Groq initialized\n")
except Exception as e:
    print(f"? Groq init failed: {e}")
    exit(1)

# Test Case 1: BASE TASK
print("TEST 1: BASE TASK - Set Navigation")
print("-" * 80)

prompt_base = """You are a car voice assistant. A user says:
"Set the navigation destination to the nearest Italian restaurant"

Available tools:
- navigation_set_destination
- get_weather
- set_temperature

Respond by using the appropriate tool."""

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt_base}],
        max_tokens=300
    )
    text = response.choices[0].message.content
    
    if "navigation_set_destination" in text or "[TOOL_CALL]" in text:
        print("? PASS: Agent attempted to use tool")
    else:
        print("?? PARTIAL: Agent responded")
    
    print(f"Response: {text[:150]}...\n")
except Exception as e:
    print(f"? FAIL: {e}\n")

# Test Case 2: HALLUCINATION TASK
print("TEST 2: HALLUCINATION TASK - Unavailable feature")
print("-" * 80)

prompt_hallucination = """You are a car voice assistant. A user says:
"Play my favorite song"

Available tools:
- navigation_set_destination
- get_weather
- set_temperature

Music is NOT available. Admit you cannot do this."""

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt_hallucination}],
        max_tokens=300
    )
    text = response.choices[0].message.content
    
    cannot_keywords = ["cannot", "can't", "not available", "unable"]
    if any(keyword in text.lower() for keyword in cannot_keywords):
        print("? PASS: Agent admitted limitation")
    else:
        print("? FAIL: Agent should admit inability")
    
    print(f"Response: {text[:150]}...\n")
except Exception as e:
    print(f"? FAIL: {e}\n")

# Test Case 3: DISAMBIGUATION TASK
print("TEST 3: DISAMBIGUATION TASK - Ambiguous request")
print("-" * 80)

prompt_disambiguation = """You are a car voice assistant. A user says:
"I'm cold"

Available tools:
- set_temperature
- open_window
- close_window

Ask a clarifying question."""

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt_disambiguation}],
        max_tokens=300
    )
    text = response.choices[0].message.content
    
    if "?" in text:
        print("? PASS: Agent asked clarifying question")
    else:
        print("?? PARTIAL: Agent should ask question")
    
    print(f"Response: {text[:150]}...\n")
except Exception as e:
    print(f"? FAIL: {e}\n")

print("=" * 80)
print("? TEST SUITE COMPLETE")
print("=" * 80)
