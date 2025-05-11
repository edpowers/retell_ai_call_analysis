import asyncio
import os
import sys

import nest_asyncio
from dotenv import load_dotenv

# Add the project root directory to Python's path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)


# Use relative imports from the package
from clients.db_call_analysis_client import CallAnalysisClient
from clients.openai_client import OpenAIClient
from clients.retell_client import RetellClient

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
