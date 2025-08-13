# main_nuke_script.py

from database.models import DatabaseManager
from database.operations import AdminOperations, QueryOperations
from config.settings import settings

db_manager = DatabaseManager(settings.DATABASE_PATH)

admin_ops = AdminOperations(db_manager)

query_ops = QueryOperations(db_manager)


def run_nuke():
    """
    Initializes the database manager and runs the nuke operation.
    """
    print("Starting database nuke operation...")

    admin_ops.nuke_database()

    print("Nuke operation script finished.")


def remove_query():
    query_ops.delete_query("3")


def show_tokens():
    print(query_ops.get_todays_total_tokens())


if __name__ == "__main__":
    # This makes the script runnable from the command line.
    # run_nuke()
    pass

# show_tokens()
