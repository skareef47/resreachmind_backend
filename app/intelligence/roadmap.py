"""
Actionable Research Roadmap Generator for ResearchMind AI.

Translates research papers and buildability insights into a concrete, 7-stage
engineering execution plan:
1. dataset preparation
2. baseline
3. model training
4. evaluation
5. backend
6. frontend integration
7. deployment

Explicitly links each milestone to supporting scholarly research papers,
datasets, and code repositories.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from app.intelligence.models import (
    ResearchRoadmap,
    RoadmapMilestone,
    RoadmapStageType,
    SupportingResource,
)
from app.schemas.paper import ResearchDocument

logger = logging.getLogger("researchmind.intelligence.roadmap")


class RoadmapGenerator:
    """
    Generates structured, research-grounded 7-stage engineering roadmaps.
    """

    def generate_roadmap(
        self,
        query: str,
        documents: List[ResearchDocument],
    ) -> ResearchRoadmap:
        """
        Synthesize research documents into an actionable 7-stage engineering roadmap.
        """
        roadmap_id = f"roadmap_{uuid.uuid4().hex[:12]}"

        # Collect resources across documents
        dataset_resources: List[SupportingResource] = []
        code_resources: List[SupportingResource] = []
        model_resources: List[SupportingResource] = []
        paper_resources: List[SupportingResource] = []

        for doc in documents:
            paper_res = SupportingResource(
                resource_type="paper",
                title=doc.title,
                url=doc.url,
                identifier=doc.doi,
                document_id=doc.internal_id,
                relevance_note=f"Core reference publication ({doc.venue or 'Scholarly Publication'}, {doc.publication_year or 'Recent'})",
            )
            paper_resources.append(paper_res)

            if doc.datasets:
                for ds in doc.datasets:
                    dataset_resources.append(
                        SupportingResource(
                            resource_type="dataset",
                            title=ds,
                            document_id=doc.internal_id,
                            relevance_note=f"Benchmark dataset cited in '{doc.title}'",
                        )
                    )

            if doc.code_url:
                code_resources.append(
                    SupportingResource(
                        resource_type="repository",
                        title=f"{doc.title} Repository",
                        url=doc.code_url,
                        document_id=doc.internal_id,
                        relevance_note=f"Verified reference codebase: {doc.code_url}",
                    )
                )

            if doc.models:
                for m in doc.models:
                    model_resources.append(
                        SupportingResource(
                            resource_type="model",
                            title=m,
                            document_id=doc.internal_id,
                            relevance_note=f"Pretrained model checkpoint cited in '{doc.title}'",
                        )
                    )

        # -------------------------------------------------------------
        # 1. Dataset Preparation
        # -------------------------------------------------------------
        ds_prep = RoadmapMilestone(
            id="ms_dataset_preparation",
            stage=RoadmapStageType.DATASET_PREPARATION,
            title="Dataset Ingestion, Cleaning & Benchmark Partitioning",
            description=(
                f"Ingest, clean, and standardize datasets for '{query}'. "
                "Construct train/validation/test splits, verify label hygiene, "
                "and implement data loader pipelines."
            ),
            prerequisites=[],
            estimated_effort="2-4 days",
            deliverables=[
                "Standardized dataset partitions (train/val/test)",
                "Schema validation and data integrity unit tests",
                "Automated data loader and preprocessing script",
            ],
            supporting_resources=dataset_resources[:4] or paper_resources[:2],
        )

        # -------------------------------------------------------------
        # 2. Baseline Establishment
        # -------------------------------------------------------------
        baseline = RoadmapMilestone(
            id="ms_baseline",
            stage=RoadmapStageType.BASELINE,
            title="Reference Baseline Implementation & Metric Calibration",
            description=(
                "Establish deterministic reference baselines (heuristic, lexical, or simpler classical models) "
                "to define the minimum acceptance threshold before training complex architectures."
            ),
            prerequisites=["ms_dataset_preparation"],
            estimated_effort="1-2 days",
            deliverables=[
                "Baseline scoring script",
                "Initial benchmark evaluation scores logged as lower bounds",
                "Inference latency profile for baseline",
            ],
            supporting_resources=paper_resources[:2],
        )

        # -------------------------------------------------------------
        # 3. Model Training
        # -------------------------------------------------------------
        model_train = RoadmapMilestone(
            id="ms_model_training",
            stage=RoadmapStageType.MODEL_TRAINING,
            title="Model Architecture Implementation & Checkpoint Training",
            description=(
                f"Implement core model architecture tailored for '{query}'. "
                "Fine-tune pretrained checkpoints or train weights with gradient clipping, "
                "learning rate schedules, and checkpoint serialization."
            ),
            prerequisites=["ms_baseline"],
            estimated_effort="3-5 days",
            deliverables=[
                "Model architecture code with modular layer definitions",
                "Training pipeline with automated checkpointing and early stopping",
                "Serialized model weights and training convergence telemetry",
            ],
            supporting_resources=(code_resources[:2] + model_resources[:2]) or paper_resources[:2],
        )

        # -------------------------------------------------------------
        # 4. Evaluation
        # -------------------------------------------------------------
        eval_ms = RoadmapMilestone(
            id="ms_evaluation",
            stage=RoadmapStageType.EVALUATION,
            title="Rigorous Benchmark Evaluation, Ablations & Error Profiling",
            description=(
                "Assess trained models on holdout benchmark partitions. "
                "Execute ablation studies to measure individual component gains, "
                "and analyze failure edge cases."
            ),
            prerequisites=["ms_model_training"],
            estimated_effort="2-3 days",
            deliverables=[
                "Comprehensive metric score card comparing trained model to baseline",
                "Ablation study summary demonstrating component utility",
                "Error analysis report documenting false positives/negatives",
            ],
            supporting_resources=paper_resources[:3],
        )

        # -------------------------------------------------------------
        # 5. Backend
        # -------------------------------------------------------------
        backend_ms = RoadmapMilestone(
            id="ms_backend",
            stage=RoadmapStageType.BACKEND,
            title="Production Service Integration & Asynchronous API Engine",
            description=(
                "Wrap verified models and ranking logic in a high-performance FastAPI service. "
                "Implement batch inference, caching layers, request validation, and telemetry."
            ),
            prerequisites=["ms_evaluation"],
            estimated_effort="2-3 days",
            deliverables=[
                "Versioned REST API endpoints under /api/v1",
                "Asynchronous inference pipeline with batching and fallback",
                "Integration test suite validating latency and schema compliance",
            ],
            supporting_resources=code_resources[:2] or paper_resources[:1],
        )

        # -------------------------------------------------------------
        # 6. Frontend Integration
        # -------------------------------------------------------------
        fe_ms = RoadmapMilestone(
            id="ms_frontend_integration",
            stage=RoadmapStageType.FRONTEND_INTEGRATION,
            title="Interactive User Dashboard & Visual Explanations",
            description=(
                "Connect frontend UI client with the backend API. "
                "Provide interactive search inputs, ranking controls, visual score breakdowns, "
                "and knowledge graph exploration."
            ),
            prerequisites=["ms_backend"],
            estimated_effort="2-4 days",
            deliverables=[
                "Responsive Next.js/React frontend with ranking and filter controls",
                "Component contribution visualization and graph viewer",
                "Client-side state management with error boundaries",
            ],
            supporting_resources=paper_resources[:2],
        )

        # -------------------------------------------------------------
        # 7. Deployment
        # -------------------------------------------------------------
        deploy_ms = RoadmapMilestone(
            id="ms_deployment",
            stage=RoadmapStageType.DEPLOYMENT,
            title="Containerization, Production Deployment & Health Monitoring",
            description=(
                "Package backend and frontend services into production Docker images. "
                "Configure cloud runtime, health monitoring, logging, and automated CI/CD."
            ),
            prerequisites=["ms_frontend_integration"],
            estimated_effort="1-2 days",
            deliverables=[
                "Multi-stage Dockerfile and container orchestration config",
                "Production environment deployment and automated health checks",
                "Logging and alerting dashboards for inference latency and errors",
            ],
            supporting_resources=code_resources[:1] or paper_resources[:1],
        )

        stages = [ds_prep, baseline, model_train, eval_ms, backend_ms, fe_ms, deploy_ms]

        return ResearchRoadmap(
            roadmap_id=roadmap_id,
            title=f"Engineering Implementation Roadmap: {query}",
            target_query=query,
            stages=stages,
            supporting_papers=[d.internal_id for d in documents],
            total_estimated_duration="2-3 weeks",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
