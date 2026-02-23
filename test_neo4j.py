from graph.neo4j_client import get_client
with get_client() as client:
    if client.verify():
        print("Connected to Neo4j successfully!")
    else:
        print("Failed to connect to Neo4j.")
