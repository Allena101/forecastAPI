from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
# Access the variables from the .env file
db_user = os.getenv("DB_USER_MONGO")
db_password = os.getenv("DB_PASSWORD_MONGO")
db_host = os.getenv("DB_HOST_MONGO")


# Function so that we can use fastAPI Depends
async def get_mongo_client():
    client = MongoClient(
        f"mongodb+srv://{db_user}:{db_password}@{db_host}/"
        f"?retryWrites=true&w=majority&appName=myMongoDB"
    )
    elmatare_db = client.elmatare_db.elmatare_collection

    yield elmatare_db
    client.close()


def individual_series(series) -> dict:
    return {
        "id": str(series["_id"]),
        "DateTime": series["DateTime"],
        "Actual": series["Actual"],
        "Forecast": series["Forecast"],
    }
