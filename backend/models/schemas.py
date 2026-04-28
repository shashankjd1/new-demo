from typing import Literal

from pydantic import BaseModel, Field


ClaimClassification = Literal["VERIFIED", "CONDITIONAL", "MISLEADING", "NON-VERIFIABLE"]
ClaimSeverity = Literal["HIGH", "MEDIUM", "LOW"]
ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


class AnalyzeRequest(BaseModel):
    product_name: str = Field(..., min_length=1)
    marketing_text: str = Field(..., min_length=1)
    specs: str = ""


class DimensionScores(BaseModel):
    transparency: int = Field(..., ge=0, le=100)
    verifiability: int = Field(..., ge=0, le=100)
    comparability: int = Field(..., ge=0, le=100)
    consistency: int = Field(..., ge=0, le=100)


class FlaggedClaim(BaseModel):
    claim: str
    classification: ClaimClassification
    severity: ClaimSeverity
    reason: str
    realistic_interpretation: str


class NormalizedSpecItem(BaseModel):
    term: str
    meaning: str


class RealWorldItem(BaseModel):
    feature: str
    insight: str
    confidence: ConfidenceLevel


class AnalysisResponse(BaseModel):
    truth_score: int = Field(..., ge=0, le=100)
    dimensions: DimensionScores
    tldr: list[str] = Field(default_factory=list, min_length=3, max_length=3)
    flagged_claims: list[FlaggedClaim] = Field(default_factory=list)
    normalized_specs: list[NormalizedSpecItem] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    real_world: list[RealWorldItem] = Field(default_factory=list)
    verdict: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: AnalysisResponse | None = None


class ChatResponse(BaseModel):
    answer: str
