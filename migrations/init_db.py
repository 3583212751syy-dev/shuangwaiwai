"""Initialize database tables and MinIO bucket (if configured).
Run: python migrations/init_db.py
"""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

S3_ENDPOINT = os.environ.get('MINIO_ENDPOINT') or os.environ.get('S3_ENDPOINT')
S3_BUCKET = os.environ.get('S3_BUCKET', 'doubleiface')
S3_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY')
S3_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY')
POSTGRES_URL = os.environ.get('POSTGRES_URL')

if POSTGRES_URL:
    engine = create_engine(POSTGRES_URL)
else:
    sqlite_path = os.path.join(os.path.dirname(__file__), '..', 'data.db')
    engine = create_engine(f'sqlite:///{sqlite_path}')

print('Connecting to DB...')
try:
    with engine.connect() as conn:
        conn.execute(text('CREATE TABLE IF NOT EXISTS tasks (id SERIAL PRIMARY KEY, request_id VARCHAR(64), input_path TEXT, output_path TEXT, s3_input_url TEXT, s3_output_url TEXT, qa_result TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, completed_at TIMESTAMP NULL);'))
        conn.commit()
    print('DB initialized (tasks table).')
except OperationalError as e:
    print('DB connection failed:', e)

# Try to create MinIO bucket
try:
    if S3_ENDPOINT and S3_ACCESS_KEY and S3_SECRET_KEY:
        import boto3
        s3 = boto3.client('s3', endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY)
        existing = [b['Name'] for b in s3.list_buckets().get('Buckets', [])]
        if S3_BUCKET not in existing:
            s3.create_bucket(Bucket=S3_BUCKET)
            print('Created bucket', S3_BUCKET)
        else:
            print('Bucket exists:', S3_BUCKET)
    else:
        print('MinIO not configured; skipping bucket creation')
except Exception as e:
    print('MinIO bucket creation skipped/failed:', e)
