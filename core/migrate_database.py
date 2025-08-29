import sqlite3
import os

def migrate_database(db_path):
    """Migrate the database to the new schema with year columns"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if year column exists in backtest_tickers
        cursor.execute("PRAGMA table_info(backtest_tickers)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'year' not in columns:
            print("Adding year column to backtest_tickers table...")
            cursor.execute("ALTER TABLE backtest_tickers ADD COLUMN year INTEGER")
            
            # Update existing rows with year values extracted from date
            cursor.execute("UPDATE backtest_tickers SET year = CAST(substr(date, 1, 4) AS INTEGER)")
            print("Updated existing rows with year values.")
        
        # Check if start_year and end_year columns exist in backtest_final_results
        cursor.execute("PRAGMA table_info(backtest_final_results)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'start_year' not in columns:
            print("Adding start_year column to backtest_final_results table...")
            cursor.execute("ALTER TABLE backtest_final_results ADD COLUMN start_year INTEGER")
        
        if 'end_year' not in columns:
            print("Adding end_year column to backtest_final_results table...")
            cursor.execute("ALTER TABLE backtest_final_results ADD COLUMN end_year INTEGER")
        
        # Update existing rows with year values
        cursor.execute("""
            UPDATE backtest_final_results 
            SET start_year = CAST(substr(start_date, 1, 4) AS INTEGER),
                end_year = CAST(substr(end_date, 1, 4) AS INTEGER)
        """)
        print("Updated existing rows with year values in backtest_final_results.")
        
        conn.commit()
        print("Database migration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    db_path = r"C:\Users\kyung\StockScanner\core\ticker_data.db"
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        print("The new schema will be created automatically when you run the main script.")
    else:
        print(f"Migrating database at {db_path}")
        migrate_database(db_path)