from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)  # Primary key 'id'
    username = Column(String(50), unique=True)  # Unique username
    password = Column(String(50))  # Storing hashed password, not unique

class Marksheet(Base):
    __tablename__ = 'marks'

    id = Column(Integer, primary_key=True, index=True)  # Primary key on 'id'
    username = Column(String(50))  # 'username' as foreign key reference, assuming relationship (can be defined)
    marks = Column(Integer)  # Marks field
    strikes = Column(Integer)  # Strikes field
