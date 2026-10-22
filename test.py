query = f"""name and query <query>{user_query}</query>, give solution."""

llm.invoke(query).content