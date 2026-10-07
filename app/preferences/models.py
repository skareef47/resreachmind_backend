from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ComputeRequirement(str, Enum):
    """Compute requirement preference levels."""

    LOW_COMPUTE = "low_compute"
    BALANCED = "balanced"
    HIGH_COMPUTE = "high_compute"


class UserPreferences(BaseModel):
    """
    User preferences across 8 core dimensions:
    1. relevance
    2. implementation relevance
    3. recency
    4. compute requirements
    5. dataset availability
    6. source-code availability
    7. pretrained model availability
    8. source quality
    """

    # 1. Relevance
    relevance_weight: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Priority for semantic & technical relevance"
    )

    # 2. Implementation relevance
    implementation_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Preference for applied systems & implementation details over pure theory",
    )

    # 3. Recency
    recency_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Preference for recently published literature"
    )

    # 4. Compute requirements
    compute_requirement: ComputeRequirement = Field(
        default=ComputeRequirement.BALANCED,
        description="Target compute envelope (low_compute for CPU/edge, balanced, or high_compute)",
    )
    compute_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Priority placed on adhering to compute requirement"
    )

    # 5. Dataset availability
    dataset_availability_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Preference for papers providing open benchmarks & datasets",
    )

    # 6. Source-code availability
    source_code_availability_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Preference for papers with verifiable open-source code repositories",
    )

    # 7. Pretrained model availability
    pretrained_model_availability_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Preference for papers offering downloadable pretrained model checkpoints",
    )

    # 8. Source quality
    source_quality_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Preference for high-impact venues and peer-reviewed citation track record",
    )

    # Optional venue & concept affinities
    preferred_venues: List[str] = Field(
        default_factory=list, description="Target publication venues (e.g., NeurIPS, ICML, CVPR, ACL)"
    )
    preferred_concepts: List[str] = Field(
        default_factory=list, description="Preferred research subfields or concepts"
    )
