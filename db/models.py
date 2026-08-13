from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Task(Base):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True)
    request_id = Column(String(64), index=True)
    input_path = Column(Text)
    output_path = Column(Text)
    s3_input_url = Column(Text)
    s3_output_url = Column(Text)
    qa_result = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
