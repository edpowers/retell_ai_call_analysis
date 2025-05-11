import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the parent directory to sys.path if it's not already there
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from clients.db_call_analysis_client import CallAnalysisClient  # noqa: E402
from db.db_model import Base  # noqa: E402


def create_database(db_path="call_analysis.db", drop_existing=False):
    """
    Create a new SQLite database with the defined schema.

    Args:
        db_path: Path to the SQLite database file
        drop_existing: If True, drop all existing tables before creating new ones

    Returns:
        SQLAlchemy session factory
    """
    # Create directory if it doesn't exist
    db_path_obj = Path(db_path).resolve()
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Create SQLite database engine
    engine = create_engine(f"sqlite:///{db_path}", echo=True)

    if drop_existing:
        # Drop all tables if they exist
        Base.metadata.drop_all(engine)
        print(f"Dropped all existing tables in {db_path}")

    # Create all tables defined in the Base metadata
    Base.metadata.create_all(engine)

    # Create a session factory
    Session = sessionmaker(bind=engine)

    print(f"Database created successfully at {db_path}")
    return Session


def ensure_call_analysis_table(db_path="call_analysis.db"):
    """
    Ensure the call_analysis table exists for the CallAnalysisClient.
    This handles the case where the table name in the client differs from the SQLAlchemy model.

    Args:
        db_path: Path to the SQLite database file
    """
    # Create a CallAnalysisClient instance
    client = CallAnalysisClient(db_path, openai_client=None)

    # Use the client's method to create the table if it doesn't exist
    client.db_client.create_table_if_not_exists()

    print(f"Ensured call_analysis table exists in {db_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Set up the call analysis database")
    parser.add_argument(
        "--db-path",
        default="data/call_analysis.db",
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--drop-existing-data",
        action="store_true",
        help="Drop all existing tables before creating new ones",
    )
    args = parser.parse_args()

    # Create the database with SQLAlchemy models
    Session = create_database(args.db_path, drop_existing=args.drop_existing_data)

    # Ensure the call_analysis table exists for the client
    ensure_call_analysis_table(args.db_path)

    # Create a session to interact with the database
    session = Session()

    # Now you can use the session to add, query, update, or delete records
    # For example:
    # from retell_ai_call_analysis.db.db_model import CallAnalysis
    # new_analysis = CallAnalysis(call_id="call123", is_analyzed=False)
    # session.add(new_analysis)
    # session.commit()

    session.close()


#    python -m retell_ai_call_analysis.db.setup_database --drop-existing-data
