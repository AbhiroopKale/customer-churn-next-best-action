"""Create the database (if missing), tables and views.  Run: python -m scripts.setup_database"""
from src import database


def main():
    database.create_database_if_missing()
    engine = database.get_engine()
    database.init_schema(engine)
    database.ping(engine)
    print("Schema + views created. Connection OK.")


if __name__ == "__main__":
    main()
