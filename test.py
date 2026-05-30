from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")
print(f"URL: '{url}'")
print(f"KEY starts with: '{key[:20]}'")

sb = create_client(url, key)
res = sb.auth.sign_up({"email": "test123@example.com", "password": "test1234"})
print(res)