# main_nuke_script.py

from database.models import DatabaseManager
from database.operations import AdminOperations
from config.settings import settings


def run_nuke():
    """
    Initializes the database manager and runs the nuke operation.
    """
    print("Starting database nuke operation...")

    # 1. Initialize the DatabaseManager with the path from your settings.
    #    This object handles the database connections.
    db_manager = DatabaseManager(settings.DATABASE_PATH)

    # 2. Initialize AdminOperations with the DatabaseManager instance.
    admin_ops = AdminOperations(db_manager)

    # 3. Now you can call the nuke_database method.
    admin_ops.nuke_database()

    print("Nuke operation script finished.")


if __name__ == "__main__":
    # This makes the script runnable from the command line.
    run_nuke()
