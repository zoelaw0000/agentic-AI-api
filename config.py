import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres.sqejnmpvkzaiwuqyqjcc:advancedataanalytic@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres"
)