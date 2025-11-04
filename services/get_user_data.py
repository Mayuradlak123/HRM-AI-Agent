from config.database import get_database  # Adjust import as per your structure
from config.logger import logger

def get_user_hr_data(user_id: str):
    """
    Fetch data from HR-related collections using the given user_id.
    - benefit_enrollments, leave_balances, payroll_records → one record per user
    - company_policies, company_announcements → latest global record (no user filter)
    Returns a list like:
    [
        {"benefit_enrollments": {...}},
        {"company_announcements": {...}},
        {"company_policies": {...}},
        {"leave_balances": {...}},
        {"payroll_records": {...}}
    ]
    """
    db = get_database()

    # Separate user-specific and global collections
    user_specific = ["benefit_enrollments", "leave_balances", "payroll_records","employees"]
    global_latest = ["company_policies", "company_announcements","company_info"]

    combined_data = []

    # Fetch user-specific collections
    for col in user_specific:
        try:
            collection = db[col]
            record = collection.find_one({"user_id": user_id}, {"_id": 0})
            if col == "employees":
                combined_data.append({"me": record or {}})
            else:
                combined_data.append({col: record or {}})
        except Exception as e:
            logger.error(f"Failed to fetch user-specific data from {col}: {e}")
            combined_data.append({col: {}})

    # Fetch global latest records
    for col in global_latest:
        try:
            collection = db[col]
            # Try sorting by created_at if available, else fallback to _id
            latest_record = collection.find_one(
                sort=[("created_at", -1)]
            ) or collection.find_one(sort=[("_id", -1)])
            
            # remove _id for clarity
            if latest_record and "_id" in latest_record:
                del latest_record["_id"]

            combined_data.append({col: latest_record or {}})
        except Exception as e:
            logger.error(f"Failed to fetch global data from {col}: {e}")
            combined_data.append({col: {}})
    return combined_data
