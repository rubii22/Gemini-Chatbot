import os
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get Gemini API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY is not set in your .env file!")

# Setup Gemini with OpenAI Agent SDK
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# Chainlit Chat Handler
@cl.on_message
async def handle_message(message: cl.Message):
    agent = Agent(
        name="Gemini-Assistant",
        instructions="You are a helpful assistant powered by Google Gemini.",
        model=model,
    )

    response = await Runner.run(agent, message.content, run_config=config)

    await cl.Message(content=response.final_output).send()
