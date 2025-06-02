import os
import sys
import time
import random
import asyncio
import chainlit as cl
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=api_key)


# Rate limiting configuration
class RateLimiter:
    def __init__(self, max_requests_per_minute=12, max_requests_per_day=1000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_day = max_requests_per_day
        self.requests_this_minute = 0
        self.requests_today = 0
        self.minute_start_time = time.time()
        self.day_start_time = time.time()

    def reset_if_needed(self):
        current_time = time.time()
        # Reset minute counter if a minute has passed
        if current_time - self.minute_start_time > 60:
            self.requests_this_minute = 0
            self.minute_start_time = current_time

        # Reset day counter if a day has passed
        if current_time - self.day_start_time > 86400:  # 86400 seconds = 24 hours
            self.requests_today = 0
            self.day_start_time = current_time

    async def wait_if_needed(self):
        self.reset_if_needed()

        # Check if we've hit rate limits
        if self.requests_this_minute >= self.max_requests_per_minute:
            wait_time = 60 - (time.time() - self.minute_start_time)
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time + 1)  # Add 1 second buffer
                self.reset_if_needed()

        if self.requests_today >= self.max_requests_per_day:
            wait_time = 86400 - (time.time() - self.day_start_time)
            if wait_time > 0:
                print(f"Daily rate limit reached. Waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time + 1)  # Add 1 second buffer
                self.reset_if_needed()

    def increment(self):
        self.reset_if_needed()
        self.requests_this_minute += 1
        self.requests_today += 1


# Create rate limiter instance
rate_limiter = RateLimiter(max_requests_per_minute=12, max_requests_per_day=1000)

# Check available models first
try:
    print("Checking available models...")
    available_models = genai.list_models()
    model_names = [model.name for model in available_models]
    print(f"Available models: {model_names}")

    # Check if any Gemini models are available
    gemini_models = [name for name in model_names if "gemini" in name.lower()]
    if not gemini_models:
        print("Error: No Gemini models available with your API key.")
        print("Please check your API key and permissions.")
        sys.exit(1)

    print(f"Available Gemini models: {gemini_models}")
except Exception as e:
    print(f"Error checking available models: {str(e)}")
    print("Continuing with default model...")

# Initialize the model with safety settings
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

# Try to find the best model to use
model_to_use = "models/gemini-2.0-flash"
try:
    # Check if our preferred model is available
    if "models/gemini-2.0-flash" in model_names:
        model_to_use = "models/gemini-2.0-flash"
    elif "models/gemini-2.0-flash-001" in model_names:
        model_to_use = "models/gemini-2.0-flash-001"
    # Fall back to other options if needed
    elif "models/gemini-1.5-flash" in model_names:
        model_to_use = "models/gemini-1.5-flash"
    elif "models/gemini-1.5-flash-latest" in model_names:
        model_to_use = "models/gemini-1.5-flash-latest"
    elif gemini_models:
        # Use the first available Gemini model
        flash_models = [m for m in gemini_models if "flash" in m.lower()]
        if flash_models:
            model_to_use = flash_models[0]
        else:
            model_to_use = gemini_models[0]
        print(f"Using alternative model: {model_to_use}")

    print(f"Initializing with model: {model_to_use}")

    model = genai.GenerativeModel(
        model_name=model_to_use,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    raise

# Use a simpler approach with a global chat
chat = None


@cl.on_chat_start
async def on_chat_start():
    global chat
    chat = model.start_chat(history=[])
    await cl.Message(
        content="Hello! I'm your Gemini AI assistant. How can I help you today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    # Declare all global variables at the beginning of the function
    global chat, model, model_to_use

    try:
        # Initialize chat if not already done
        if chat is None:
            chat = model.start_chat(history=[])

        # Send a thinking message
        thinking_msg = cl.Message(content="Thinking...")
        thinking_msg = await thinking_msg.send()

        # Wait if needed to respect rate limits
        await rate_limiter.wait_if_needed()

        # Generate a response with retry logic for rate limiting
        max_retries = 5
        retry_count = 0
        backoff_time = 1

        while retry_count < max_retries:
            try:
                # Try to send message
                response = chat.send_message(message.content)

                # If successful, increment rate limiter and break
                rate_limiter.increment()

                # Update the thinking message with the response
                # Create a new message with the response content instead of updating
                await thinking_msg.remove()
                await cl.Message(content=response.text).send()
                break

            except ResourceExhausted as e:
                # This is a rate limit error (429)
                retry_count += 1
                if retry_count >= max_retries:
                    raise

                # Parse retry delay from error message if available
                retry_delay = backoff_time
                error_str = str(e)
                if "retry_delay" in error_str and "seconds" in error_str:
                    try:
                        # Try to extract the retry delay from the error message
                        retry_delay_index = error_str.find("retry_delay")
                        seconds_index = error_str.find("seconds", retry_delay_index)
                        retry_delay_str = error_str[retry_delay_index:seconds_index]
                        # Find the number in the string
                        import re

                        numbers = re.findall(r"\d+", retry_delay_str)
                        if numbers:
                            retry_delay = int(numbers[0])
                    except Exception:
                        # If parsing fails, use exponential backoff
                        retry_delay = backoff_time

                # Add some jitter to prevent all clients from retrying at the same time
                jitter = random.uniform(0, 0.5)
                wait_time = retry_delay + jitter

                # Update thinking message to inform user
                # Remove the old thinking message and send a new one
                await thinking_msg.remove()
                thinking_msg = await cl.Message(
                    content=f"Rate limit exceeded. Retrying in {wait_time:.1f} seconds... (Attempt {retry_count}/{max_retries})"
                ).send()

                # Wait before retrying
                await asyncio.sleep(wait_time)

                # Exponential backoff for next retry if needed
                backoff_time *= 2

    except Exception as e:
        error_message = str(e)
        advice = ""

        # Provide helpful advice for specific errors
        if "quota" in error_message.lower() or "429" in error_message:
            advice = (
                "\n\nYou've hit API rate limits. The bot will automatically retry with backoff. If this persists, consider:\n"
                + "1. Waiting a few minutes before trying again\n"
                + "2. Reducing the frequency of your requests\n"
                + "3. Upgrading your Google AI Studio plan for higher quotas"
            )
        elif "not found" in error_message.lower() or "404" in error_message:
            advice = "\n\nThe specified model may not be available. The system will attempt to use an alternative model."
            # Try to switch to another model
            await cl.Message(
                content="Attempting to switch to an alternative model..."
            ).send()
            try:
                alt_models = [
                    "models/gemini-2.0-flash-001",
                    "models/gemini-1.5-flash",
                    "models/gemini-1.5-flash-latest",
                ]
                for alt_model in alt_models:
                    if alt_model in model_names and alt_model != model_to_use:
                        model_to_use = alt_model
                        model = genai.GenerativeModel(
                            model_name=model_to_use,
                            generation_config=generation_config,
                            safety_settings=safety_settings,
                        )
                        chat = model.start_chat(history=[])
                        await cl.Message(
                            content=f"Switched to model: {model_to_use}"
                        ).send()
                        return await main(message) 
            except Exception as model_switch_error:
                advice += f"\nFailed to switch models: {str(model_switch_error)}"

        await cl.Message(content=f"Error: {error_message}{advice}").send()