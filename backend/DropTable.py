import mysql.connector
def drop_all_tables(database_name, user, password, host='localhost'):
    connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database_name
    )
    cursor = connection.cursor()

    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
    for (table_name,) in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")

    cursor.close()
    connection.close()

def run_sql_script(script_path, user, password, host='localhost', database=None):
    connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = connection.cursor()

    with open(script_path, 'r') as file:
        sql_script = file.read()

    for statement in sql_script.split(';'):
        if statement.strip():
            cursor.execute(statement)

    connection.commit()
    cursor.close()
    connection.close()

def Drop_Table():
    script_path = 'database/db-schema.sql'
    user='root'
    password='pukkaj'
    database_name='hip_fracture_study'
    drop_all_tables(database_name, user, password)
    run_sql_script(script_path, user, password, database=database_name)

