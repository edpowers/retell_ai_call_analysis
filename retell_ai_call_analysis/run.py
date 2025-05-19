import argparse
import asyncio
import sys
from pathlib import Path

import nest_asyncio
from dotenv import load_dotenv

# Add the project root directory to Python's path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# Use relative imports from the package
from clients.db_call_analysis_client import CallAnalysisClient  # noqa: E402
from clients.openai_client import OpenAIClient  # noqa: E402
from clients.retell_client import RetellClient  # noqa: E402

nest_asyncio.apply()


async def main(days_ago: int = 1) -> None:
    load_dotenv()
    retell_client = RetellClient()
    call_responses = retell_client.get_calls(days_ago=days_ago)

    # Initialize OpenAI client
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
    parser = argparse.ArgumentParser(description="Process Retell AI calls for analysis")
    parser.add_argument(
        "--days-ago",
        type=int,
        default=1,
        help="Number of days to look back for calls (default: 1)",
    )

    args = parser.parse_args()
    asyncio.run(main(days_ago=args.days_ago))
