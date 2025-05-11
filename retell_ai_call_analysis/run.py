import asyncio

import nest_asyncio

# # Add the project root directory to Python's path
# if not os.path.exists(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.db_call_analysis_client import CallAnalysisClient
from clients.openai_client import OpenAIClient
from clients.retell_client import RetellClient
from dotenv import load_dotenv

nest_asyncio.apply()


async def main():
    load_dotenv()
    retell_client = RetellClient()
    call_responses = retell_client.get_calls(days_ago=1)

    # Initialize OpenAI client (missing in original code)
    openai_client = OpenAIClient()

    call_analysis_client = CallAnalysisClient(
        db_path="call_analysis.db", openai_client=openai_client
    )

    call_responses_filtered = call_analysis_client.db_client.filter_new_calls(
        call_responses
    )

    await call_analysis_client.process_calls_and_notify(
        call_responses_filtered,
        show_verbose_output=False,
        ignore_existing_calls=False,
    )


if __name__ == "__main__":
    asyncio.run(main())
