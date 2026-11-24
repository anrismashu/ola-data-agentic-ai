import psycopg2
import os 

class DataBaseUtils:

    def __init__(self, db_config):
        self.db_config = {
            **db_config,
            "sslmode": "require",
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }
        if os.getenv("hostaddr"):
            self.db_config["hostaddr"] = os.getenv("hostaddr")
        self.connection = None
        try:
            self._connect()
        except Exception as e:
            print(f"Error connecting to database: {e}")

    def _connect(self):
        try:
            if self.connection and self.connection.closed == 0:
                self.connection.close()
        except Exception:
            pass
        self.connection = psycopg2.connect(**self.db_config)

    def schema_detils(self, schema_name: str = "public"):
        cursor = None
        schema_info_context = f"Database_Schema: {schema_name}\n"
        try:
            if not self.connection or self.connection.closed:
                self.connection = psycopg2.connect(**self.db_config)

            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
                (schema_name,),
            )
            table_list = cursor.fetchall()

            for table in table_list:
                table_name = table[0]
                schema_info_context += f"\nTable: {table_name}\n"

                cursor.execute(
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;",
                    (table_name,),
                )
                columns = cursor.fetchall()
                for column in columns:
                    schema_info_context += (
                        f"Column: {column[0]}, Data Type: {column[1]}\n"
                    )

                cursor.execute(
                    f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT 5;'
                )
                sample_data = cursor.fetchall()
                schema_info_context += "Sample Data:\n"
                for row in sample_data:
                    schema_info_context += f"{row}\n"

        except Exception as e:
            print(f"Error retrieving schema details: {e}")
            return ""
        finally:
            if cursor:
                cursor.close()
        return schema_info_context

    # Fixed indentation: this belongs INSIDE the class
    def execute_query(self, query: str) -> str:
        for attempt in range(2):
            try:
                if self.connection is None or self.connection.closed != 0:
                    self._connect()

                with self.connection.cursor() as cursor:
                    cursor.execute(query)
                    if cursor.description is None:
                        return "Query executed successfully."
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()

                if not rows:
                    return "Query executed successfully, but returned 0 rows."

                # Include column names so the final-answer LLM never misreads columns
                return str([dict(zip(columns, row)) for row in rows])

            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                if attempt == 0:
                    self._connect()
                    continue
                return f"Database connection error: {e}"

            except Exception as e:
                if self.connection and self.connection.closed == 0:
                    self.connection.rollback()
                return f"Query execution error: {e}"