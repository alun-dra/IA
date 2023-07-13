import psycopg2

def create_connection():
    try:
        connection = psycopg2.connect(
            host="containers-us-west-130.railway.app",
            port="6892",
            database="railway",
            user="postgres",
            password="5hwnvWaYx4ULLvmuzijE"
        )
        return connection
    except (Exception, psycopg2.Error) as error:
        print("Error al conectar a la base de datos:", error)

def close_connection(connection):
    if connection:
        connection.close()
        print("Conexión cerrada.")