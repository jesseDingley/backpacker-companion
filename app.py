import os
import sys
from dotenv import load_dotenv

def is_production_env() -> bool:
    """Returns True is we are in a production env."""
    load_dotenv()
    return os.environ.get("ENV") != "dev"

def sqlite_setup() -> None:
    """Setups sqlite for streamlit."""
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def main() -> None:
    """Main."""
    if is_production_env():
        sqlite_setup()
    from backend.core import run
    run()

if __name__ == "__main__":
    main()
