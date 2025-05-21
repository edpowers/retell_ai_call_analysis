from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class CallAnalysis(Base):
    """Database model for storing call analysis results."""

    __tablename__ = "call_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(255), unique=True, nullable=False, index=True)
    start_timestamp = Column(Integer, nullable=True)  # Millisecond timestamp
    timestamp = Column(DateTime, nullable=True)
    call_url = Column(String(512), nullable=True)
    is_analyzed = Column(Boolean, default=False, nullable=True)
    has_issue = Column(Boolean, default=False, nullable=True)
    needs_human_review = Column(Boolean, default=False, nullable=True)
    transcript = Column(Text, nullable=True)  # Using Text for very long strings
    analysis = Column(Text, nullable=True)  # Using Text for very long strings
    label = Column(String(100), nullable=True)  # Basic string for label

    # Detailed analysis fields
    issue_type = Column(String(100), nullable=True)
    issue_description = Column(Text, nullable=True)
    success_status = Column(String(50), nullable=True)
    contact_info_captured = Column(Boolean, default=False, nullable=True)
    booking_attempted = Column(Boolean, default=False, nullable=True)
    booking_successful = Column(Boolean, default=False, nullable=True)
    ai_detection_question = Column(Boolean, default=False, nullable=True)
    hang_up_early = Column(Boolean, default=False, nullable=True)
    dynamic_var_mismatch = Column(Boolean, default=False, nullable=True)
    notes = Column(Text, nullable=True)
    dynamic_variables = Column(JSON, nullable=True)

    # Duration in milliseconds
    duration_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=True
    )

    def __repr__(self):
        return f"<CallAnalysis(call_id='{self.call_id}', is_analyzed={self.is_analyzed}, has_issue={self.has_issue})>"
