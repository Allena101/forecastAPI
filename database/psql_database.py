from sqlmodel import SQLModel, Session, create_engine
from dotenv import load_dotenv
import os


load_dotenv()

## Old local method
# DB_USER_PSQL = os.getenv("DB_USER_PSQL")
# DB_PASSWORD_PSQL = os.getenv("DB_PASSWORD_PSQL")
# DB_HOST_PSQL = os.getenv("DB_HOST_PSQL")
# DB_PORT_PSQL = os.getenv("DB_PORT_PSQL")
# DB_NAME_PSQL = os.getenv("DB_NAME_PSQL")
# URL_DATABASE = f"postgresql://{DB_USER_PSQL}:{DB_PASSWORD_PSQL}@{DB_HOST_PSQL}:{DB_PORT_PSQL}/{DB_NAME_PSQL}"
# engine = create_engine(URL_DATABASE)

# From Render venv file
URL_PSQL = os.getenv("URL_PSQL")
engine = create_engine(URL_PSQL)


def get_session():
    with Session(engine) as session:
        yield session


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


create_db_and_tables()
