from sqlalchemy import create_engine, text

# In-memory SQLite: no external database server required :)
engine = create_engine("sqlite:///:memory:")

# Populate a toy database.
with engine.begin() as connection:
    connection.execute(text("""
        CREATE TABLE production (
            order_id TEXT,
            good_qty INTEGER
        )
    """))
    connection.execute(
        text("INSERT INTO production VALUES (:order_id, :good_qty)"),
        [
            {"order_id": "MO1", "good_qty": 100},
            {"order_id": "MO2", "good_qty": 80},
        ],
    )

# Query with a bound parameter.
with engine.connect() as connection:
    total = connection.execute(
        text("""
            SELECT SUM(good_qty)
            FROM production
            WHERE good_qty >= :minimum
        """),
        {"minimum": 90},
    ).scalar_one() # ask for 1 result only

print(total)  # 100
engine.dispose()