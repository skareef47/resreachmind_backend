"""
Pydantic schemas for the Paper Intelligence Engine in ResearchMind AI.

Provides structured models for:
- Paper Summary & Research Problem
- Verified/Inferred Datasets, Methodology, and Models
- Evidence-grounded Limitations (EXPLICIT, DERIVED, UNKNOWN)
- Mitigations with strict provenance (AUTHOR-PROPOSED vs RESEARCHMIND-PROPOSED)
- Research Gaps and Actionable Opportunities
- Provenance Audit Trail
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LimitationType(str, Enum):
    """Grounding category of a paper limitation."""

    EXPLICIT = "EXPLICIT"    # Directly stated by the paper authors
    DERIVED = "DERIVED"      # Reasonably inferred from methodology/evaluation
    UNKNOWN = "UNKNOWN"      # Insufficient evidence in available content


class EvidenceStatus(str, Enum):
    """Grounding verification level of an extraction."""

    VERIFIED = "VERIFIED"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class ProposalType(str, Enum):
    """Distinguishes recommendation attribution."""

    AUTHOR_PROPOSED = "AUTHOR-PROPOSED"
    RESEARCHMIND_PROPOSED = "RESEARCHMIND-PROPOSED"


class PaperLimitationItem(BaseModel):
    """Individual limitation grounded in paper evidence."""

    limitation_id: str = Field(..., description="Unique limitation identifier (e.g. lim_001)")
    text: str = Field(..., description="Concise statement of the limitation")
    type: LimitationType = Field(..., description="Limitation classification: EXPLICIT, DERIVED, UNKNOWN")
    evidence: str = Field(..., description="Supporting excerpt or methodology rationale")
    evidence_source: str = Field(..., description="Source location: abstract, methodology, evaluation, discussion, etc.")
    confidence: float = Field(default=0.85, description="Confidence score in [0.0, 1.0]")


class PaperMitigationItem(BaseModel):
    """Actionable recommendation to overcome a specific limitation."""

    mitigation_id: str = Field(..., description="Unique mitigation identifier (e.g. mit_001)")
    limitation_id: str = Field(..., description="Associated limitation identifier")
    limitation_text: str = Field(..., description="Text of the limitation being addressed")
    proposal_type: ProposalType = Field(..., description="AUTHOR-PROPOSED or RESEARCHMIND-PROPOSED")
    recommendation: str = Field(..., description="Concrete actionable recommendation")
    rationale: str = Field(..., description="Theoretical or engineering justification")


class ResearchGapItem(BaseModel):
    """Evidence-grounded research gap emerging from the paper."""

    gap_id: str = Field(..., description="Unique gap identifier (e.g. gap_001)")
    description: str = Field(..., description="Grounded statement of what is unaddressed or unexplored")
    evidence_status: EvidenceStatus = Field(default=EvidenceStatus.INFERRED, description="VERIFIED, INFERRED, UNKNOWN")
    derived_from: List[str] = Field(default_factory=list, description="Components informing gap: limitations, evaluation, dataset, etc.")


class ResearchOpportunityItem(BaseModel):
    """Actionable follow-on research opportunity."""

    opportunity_id: str = Field(..., description="Unique opportunity identifier (e.g. opp_001)")
    title: str = Field(..., description="Concise research opportunity title")
    description: str = Field(..., description="Detailed research direction")
    actionable_steps: List[str] = Field(default_factory=list, description="Concrete engineering or experimental steps")
    expected_impact: str = Field(..., description="Anticipated impact or performance gain")


class ProvenanceEvidenceItem(BaseModel):
    """Traceable evidence item linking claims to source metadata."""

    claim: str = Field(..., description="Claim or extraction statement")
    status: EvidenceStatus = Field(..., description="VERIFIED, INFERRED, or UNKNOWN")
    source_field: str = Field(..., description="Field providing evidence: title, abstract, datasets, methods, etc.")
    citation: str = Field(..., description="Source identifier or paper reference")


class ExtractionComponentItem(BaseModel):
    """Individual extracted entity for dataset, methodology, model, or results."""

    name: str = Field(..., description="Name or identifier of the component")
    details: str = Field(..., description="Descriptive context or specification")
    status: EvidenceStatus = Field(..., description="VERIFIED, INFERRED, UNKNOWN")
    source: str = Field(default="paper_content", description="Provenance location")


class PaperIntelligenceRequest(BaseModel):
    """Request payload for POST /api/v1/paper-intelligence."""

    paper_id: Optional[str] = Field(default=None, description="Internal ID or source ID of the target paper")
    paper: Optional[Dict[str, Any]] = Field(default=None, description="Full or partial ResearchDocument dictionary")


class PaperIntelligenceResponse(BaseModel):
    """Complete Paper Intelligence analysis for a single research paper."""

    paper_id: str = Field(..., description="Unique paper identifier")
    title: str = Field(..., description="Paper title")
    summary: str = Field(..., description="Evidence-grounded comprehensive summary")
    research_problem: str = Field(..., description="Target problem or challenge addressed by the paper")
    dataset: List[ExtractionComponentItem] = Field(default_factory=list, description="Dataset specifications")
    methodology: List[ExtractionComponentItem] = Field(default_factory=list, description="Methodological procedures")
    model: List[ExtractionComponentItem] = Field(default_factory=list, description="Model architectures and backbones")
    results: List[ExtractionComponentItem] = Field(default_factory=list, description="Key reported empirical findings")
    limitations: List[PaperLimitationItem] = Field(default_factory=list, description="Itemized limitations with type")
    mitigations: List[PaperMitigationItem] = Field(default_factory=list, description="How to overcome limitations")
    research_gaps: List[ResearchGapItem] = Field(default_factory=list, description="Grounded research gaps")
    opportunities: List[ResearchOpportunityItem] = Field(default_factory=list, description="Actionable research opportunities")
    evidence: List[ProvenanceEvidenceItem] = Field(default_factory=list, description="Traceable claim provenance records")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
