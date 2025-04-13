from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Create SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./loan_applications.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define database models
class Application(Base):
    __tablename__ = "applications"
    
    id = Column(Integer, primary_key=True, index=True)
    no_of_dependents = Column(Integer)
    education = Column(String)
    self_employed = Column(String)
    income_annum = Column(Float)
    loan_amount = Column(Float)
    loan_term = Column(Integer)
    cibil_score = Column(Integer)
    residential_assets_value = Column(Float)
    commercial_assets_value = Column(Float)
    luxury_assets_value = Column(Float)
    bank_asset_value = Column(Float)
    
    predictions = relationship("Prediction", back_populates="application")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"))
    prediction = Column(Boolean)
    probability = Column(Float)
    explanation = Column(Text)
    feature_importance = Column(Text)  # Stored as JSON string
    
    application = relationship("Application", back_populates="predictions")

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()