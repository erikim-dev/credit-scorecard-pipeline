"""
Pydantic schemas for the Credit Risk Scoring API.
"""

from pydantic import BaseModel, Field


class ApplicationInput(BaseModel):
    """Input schema — mirrors key features from the application table."""

    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    income: float = Field(..., gt=0, description="Annual income")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    annuity: float = Field(..., gt=0, description="Monthly annuity payment")
    goods_price: float = Field(0, ge=0, description="Price of goods for consumer loans")

    employment_years: float = Field(0, ge=0, description="Years of employment")
    education_type: str = Field("Secondary", description="Education level")
    family_status: str = Field("Married", description="Marital status")
    housing_type: str = Field("House / apartment", description="Housing situation")

    bureau_loan_count: int = Field(0, ge=0, description="Number of bureau credit lines")
    active_credits: int = Field(0, ge=0, description="Currently active credit lines")
    total_debt: float = Field(0, ge=0, description="Total outstanding debt from bureau")
    overdue_count: int = Field(0, ge=0, description="Number of overdue bureau credits")

    ext_source_1: float = Field(None, description="External data source score 1")
    ext_source_2: float = Field(None, description="External data source score 2")
    ext_source_3: float = Field(None, description="External data source score 3")


class ScoringResult(BaseModel):
    """API response schema."""

    default_probability: float = Field(..., description="Predicted probability of default")
    credit_score: int = Field(..., description="Mapped credit score (300–850 scale)")
    risk_tier: str = Field(..., description="Low / Medium / High")
    recommendation: str = Field(..., description="Approve / Review / Decline")
    top_risk_factors: list[dict] = Field(
        default_factory=list,
        description="Top SHAP-based risk drivers",
    )
    model_version: str = Field("1.0.0", description="Model version used for scoring")


class BatchInput(BaseModel):
    """Batch scoring request."""
    applications: list[ApplicationInput] = Field(
        ..., min_length=1, max_length=1000,
        description="List of applications to score",
    )


class BatchResult(BaseModel):
    """Batch scoring response."""
    results: list[ScoringResult]
    total: int


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_name: str
    model_version: str
    model_type: str
    metrics: dict
    features_count: int
    training_date: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
