from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class CallAnalysis(Base):
    """Database model for storing call analysis results."""

    __tablename__ = "call_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(255), unique=True, nullable=False, index=True)
    start_timestamp = Column(Integer, nullable=True)  # Millisecond timestamp
    recording_url = Column(String(512), nullable=True)
    is_analyzed = Column(Boolean, default=False, nullable=False)
    has_issue = Column(Boolean, default=False, nullable=False)
    needs_human_review = Column(Boolean, default=False, nullable=False)
    transcript = Column(Text, nullable=True)  # Using Text for very long strings
    analysis = Column(Text, nullable=True)  # Using Text for very long strings
    label = Column(String(100), nullable=True)  # Basic string for label

    # Detailed analysis fields
    issue_type = Column(String(100), nullable=True)
    issue_description = Column(Text, nullable=True)
    success_status = Column(String(50), nullable=True)
    contact_info_captured = Column(Boolean, default=False, nullable=False)
    booking_attempted = Column(Boolean, default=False, nullable=False)
    booking_successful = Column(Boolean, default=False, nullable=False)
    ai_detection_question = Column(Boolean, default=False, nullable=False)
    hang_up_early = Column(Boolean, default=False, nullable=False)
    dynamic_var_mismatch = Column(Boolean, default=False, nullable=False)
    notes = Column(Text, nullable=True)

    # Duration in milliseconds
    duration_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self):
        return f"<CallAnalysis(call_id='{self.call_id}', is_analyzed={self.is_analyzed}, has_issue={self.has_issue})>"
