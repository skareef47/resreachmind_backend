"""
Actionable, Architecture-Grounded Research Implementation Roadmap Generator for ResearchMind AI.

Converts selected research documents, extracted implementation evidence,
hardware profiles, software dependencies, user constraints, and synthesized
technical architecture directly into an actionable, concrete 8-phase engineering roadmap:

Phase 1 — Research Specification & Benchmark Dataset Acquisition
Phase 2 — Computational Runtime & Software Stack Setup
Phase 3 — Data Preparation & Preprocessing Pipeline
Phase 4 — Model Architecture & Backbone Implementation
Phase 5 — Classifier Head & Model Training Formulation
Phase 6 — Empirical Evaluation & Benchmark Validation
Phase 7 — Inference Optimization & Explainability
Phase 8 — Serving, API & Containerized Deployment

Rigorous evidence grounding:
- Roadmap tasks are generated directly from the actual synthesized architecture components!
- Each task contains: phase, task, description, dependency, input, output, technology, difficulty, evidence_status, supporting_papers.
- Distinguishes verified paper facts from inferred engineering scaffold.
- If evidence is unavailable, explicitly labels: 'Evidence insufficient — generic implementation scaffold'.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.intelligence.architecture import ArchitectureGenerator
from app.intelligence.models import (
    ArchitectureComponent,
    EvidenceStatus,
    ResearchArchitectureResponse,
    ResearchRoadmap,
    ResearchRoadmapResponse,
    RoadmapDependencyEdge,
    RoadmapMilestone,
    RoadmapPhase,
    RoadmapStageType,
    RoadmapTask,
    SupportingResource,
    TechnicalArchitecture,
)
from app.schemas.paper import ResearchDocument

logger = logging.getLogger("researchmind.intelligence.roadmap")


def compute_architecture_hash(
    components: List[Any],
    technologies: Optional[List[str]] = None,
    dependencies: Optional[List[Any]] = None,
    paper_ids: Optional[List[str]] = None,
) -> str:
    """
    Computes a deterministic SHA256 fingerprint for a technical architecture.
    Normalizes architecture components, technologies, dependencies, and paper IDs.
    """
    comp_tuples = []
    for c in components or []:
        if isinstance(c, dict):
            c_name = str(c.get("component", "")).strip().lower()
            c_val = str(c.get("value", "")).strip().lower()
        else:
            c_name = str(getattr(c, "component", "")).strip().lower()
            c_val = str(getattr(c, "value", "")).strip().lower()
        if c_name or c_val:
            comp_tuples.append((c_name, c_val))
    comp_tuples.sort()

    tech_set = set()
    if technologies:
        for t in technologies:
            if t:
                tech_set.add(str(t).strip().lower())
    tech_list = sorted(list(tech_set))

    dep_set = set()
    if dependencies:
        for d in dependencies:
            if isinstance(d, dict):
                d_name = str(d.get("name", "")).strip().lower()
            else:
                d_name = str(getattr(d, "name", str(d))).strip().lower()
            if d_name:
                dep_set.add(d_name)
    dep_list = sorted(list(dep_set))

    pid_set = set()
    if paper_ids:
        for p in paper_ids:
            if p:
                pid_set.add(str(p).strip().lower())
    pid_list = sorted(list(pid_set))

    raw_payload = json.dumps(
        {
            "components": comp_tuples,
            "dependencies": dep_list,
            "paper_ids": pid_list,
            "technologies": tech_list,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()


class RoadmapGenerator:
    """
    Generates structured, paper-grounded 8-phase implementation roadmaps directly from architecture.
    """

    def __init__(self, arch_generator: Optional[ArchitectureGenerator] = None) -> None:
        self.arch_generator = arch_generator or ArchitectureGenerator()

    def generate_roadmap(
        self,
        documents: Optional[List[ResearchDocument]] = None,
        architecture: Optional[Any] = None,
        constraints: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        selection_mode: Optional[str] = "optimized",
        **kwargs,
    ) -> ResearchRoadmapResponse:
        """
        Synthesize selected research papers and technical architecture into an actionable roadmap.
        """
        if documents is None and "documents" in kwargs:
            documents = kwargs["documents"]
        documents = documents or []
        if query is None and "query" in kwargs:
            query = kwargs["query"]
        constraints = constraints or {}
        roadmap_id = f"roadmap_{uuid.uuid4().hex[:12]}"

        # Resolve primary title & query
        if not query:
            if documents:
                query = documents[0].title
            else:
                query = "Research System Implementation"

        target_query = query
        primary_doc = documents[0] if documents else None
        primary_title = primary_doc.title if primary_doc else "Selected Research"
        selected_paper_ids = [d.internal_id for d in documents if d.internal_id]

        # -----------------------------------------------------------------
        # 1. Ensure Technical Architecture Exists (Synthesize if missing)
        # -----------------------------------------------------------------
        arch_response: Optional[ResearchArchitectureResponse] = None
        arch_components: Dict[str, ArchitectureComponent] = {}
        arch_hardware = []
        arch_dependencies = []
        is_scaffold = False

        if architecture:
            if isinstance(architecture, ResearchArchitectureResponse):
                arch_response = architecture
                for c in arch_response.architecture.components:
                    arch_components[c.component.lower()] = c
                arch_hardware = arch_response.architecture.hardware
                arch_dependencies = arch_response.architecture.dependencies
                is_scaffold = arch_response.is_generic_scaffold
            elif isinstance(architecture, dict):
                comp_list = architecture.get("components") or architecture.get("architecture", {}).get("components", [])
                for c in comp_list:
                    if isinstance(c, dict) and "component" in c:
                        ac = ArchitectureComponent(**c)
                        arch_components[ac.component.lower()] = ac
                arch_hardware = architecture.get("hardware") or architecture.get("architecture", {}).get("hardware", [])
                arch_dependencies = architecture.get("dependencies") or architecture.get("architecture", {}).get("dependencies", [])
                is_scaffold = architecture.get("is_generic_scaffold", False)
            elif hasattr(architecture, "components"):
                for c in architecture.components:
                    arch_components[c.component.lower()] = c
                arch_hardware = getattr(architecture, "hardware", [])
                arch_dependencies = getattr(architecture, "dependencies", [])
        elif documents:
            arch_response = self.arch_generator.generate_architecture(
                documents=documents,
                constraints=constraints,
                query=query,
                selection_mode=selection_mode,
            )
            for c in arch_response.architecture.components:
                arch_components[c.component.lower()] = c
            arch_hardware = arch_response.architecture.hardware
            arch_dependencies = arch_response.architecture.dependencies
            is_scaffold = arch_response.is_generic_scaffold
        else:
            is_scaffold = True

        comp_dataset = arch_components.get("dataset")
        comp_preproc = arch_components.get("preprocessing")
        comp_feat = arch_components.get("feature extraction")
        comp_model = arch_components.get("model")
        comp_train = arch_components.get("training")
        comp_eval = arch_components.get("evaluation")
        comp_infer = arch_components.get("inference")
        comp_backend = arch_components.get("backend/api")
        comp_frontend = arch_components.get("frontend")
        comp_deploy = arch_components.get("deployment")
        comp_explainability = arch_components.get("explainability")

        # Gather all technologies across components and dependencies
        arch_technologies = []
        for c in arch_components.values():
            if hasattr(c, "technologies") and c.technologies:
                arch_technologies.extend(c.technologies)
        for d in arch_dependencies:
            d_name = d.get("name") if isinstance(d, dict) else getattr(d, "name", str(d))
            if d_name:
                arch_technologies.append(d_name)

        arch_hash = compute_architecture_hash(
            components=list(arch_components.values()),
            technologies=arch_technologies,
            dependencies=arch_dependencies,
            paper_ids=selected_paper_ids,
        )

        logger.info(
            "ROADMAP INPUT: query=%s, papers=%s, architecture_components=%s, architecture_technologies=%s, architecture_hash=%s",
            target_query,
            selected_paper_ids,
            [(k, v.value) for k, v in arch_components.items()],
            sorted(list(set(arch_technologies))),
            arch_hash,
        )

        # -----------------------------------------------------------------
        # 2. Extract Document Metadata Evidence
        # -----------------------------------------------------------------
        verified_code_urls = [d.code_url for d in documents if d.code_url]
        primary_paper_name = primary_title

        # Check buildability breakdown for hardware
        has_verified_hw = False
        hw_ev_str = ""
        for doc in documents:
            bb = getattr(doc, "buildability_breakdown", {}) or {}
            hw_info = bb.get("hardware")
            if hw_info and isinstance(hw_info, dict) and hw_info.get("status") == "verified":
                has_verified_hw = True
                if hw_info.get("evidence"):
                    hw_ev_str = hw_info["evidence"]

        # Hardware & VRAM constraints
        max_vram_gb = float(constraints.get("max_gpu_memory_gb", 24))
        is_limited_gpu = bool(constraints.get("limited_gpu") or max_vram_gb <= 8)
        is_limited_time = bool(constraints.get("limited_time") or constraints.get("time_budget_weeks", 4) <= 1)
        public_datasets_only = bool(constraints.get("public_datasets_only", False))
        open_source_only = bool(constraints.get("open_source_only", False))
        target_env = str(constraints.get("target_environment", "cloud")).lower()

        # =================================================================
        # PHASE 1 — Research Specification & Benchmark Dataset Acquisition
        # =================================================================
        p1_tasks: List[RoadmapTask] = []
        p1_evidence: List[Dict[str, Any]] = []

        ds_val = comp_dataset.value if comp_dataset else "Domain Research Benchmark Dataset"
        has_dataset = bool(
            comp_dataset
            and "unknown" not in comp_dataset.value.lower()
            and comp_dataset.value != "Domain Research Benchmark Dataset"
        )
        ds_status_str = comp_dataset.evidence_status.value if comp_dataset else ("verified" if has_dataset else "unknown")

        p1_tasks.append(
            RoadmapTask(
                phase=1,
                task=f"Review theoretical formulations and methodology of '{primary_paper_name}'",
                description="Synthesize key algorithmic claims, equations, and experimental setups from manuscript.",
                dependency="None",
                input="Selected research paper manuscript",
                output="Theoretical methodology breakdown document",
                technology="Scholarly Analysis",
                difficulty="Low",
                evidence_status="VERIFIED" if not is_scaffold else "UNKNOWN",
                evidence_source=primary_paper_name,
                supporting_papers=[primary_paper_name],
            )
        )
        p1_tasks.append(
            RoadmapTask(
                phase=1,
                task=f"Define mathematical baseline acceptance criteria and target metric thresholds for '{primary_paper_name}'",
                description=f"Extract reported benchmark performance figures from '{primary_paper_name}' to set implementation thresholds.",
                dependency="None",
                input="Published manuscript performance scorecard",
                output="Formal acceptance metric thresholds specification",
                technology="Specification Document",
                difficulty="Low",
                evidence_status="VERIFIED" if not is_scaffold else "UNKNOWN",
                evidence_source=primary_paper_name,
                supporting_papers=[primary_paper_name],
            )
        )

        if has_dataset:
            p1_tasks.append(
                RoadmapTask(
                    phase=1,
                    task=f"Acquire and prepare empirical benchmark dataset: {ds_val}",
                    description=f"Source, download, and verify integrity of dataset '{ds_val}'. Evidence: {comp_dataset.evidence if comp_dataset else 'Benchmark specification'}.",
                    dependency="None",
                    input="Public benchmark archives / data providers",
                    output=f"Raw dataset archive files ({ds_val})",
                    technology="huggingface / kaggle api / wget",
                    difficulty="Low",
                    evidence_status=ds_status_str.upper(),
                    evidence_source=comp_dataset.evidence_source if comp_dataset else primary_paper_name,
                    supporting_papers=[primary_paper_name],
                )
            )
            p1_tasks.append(
                RoadmapTask(
                    phase=1,
                    task=f"Validate train/validation/test split counts and sample balance for {ds_val}",
                    description="Verify sample counts, patient/class stratifications, and holdout splits to ensure exact reproducibility with published tables.",
                    dependency=f"Acquire and prepare empirical benchmark dataset: {ds_val}",
                    input="Raw benchmark dataset partitions",
                    output="Partition audit report with verified class balance and slice counts",
                    technology="python / pandas / numpy",
                    difficulty="Low",
                    evidence_status=ds_status_str.upper(),
                    evidence_source=comp_dataset.evidence_source if comp_dataset else primary_paper_name,
                    supporting_papers=[primary_paper_name],
                )
            )
            p1_evidence.append({
                "component": "Dataset",
                "value": ds_val,
                "source_paper": primary_paper_name,
                "evidence": f"Named benchmark datasets extracted from research metadata: {ds_val}",
                "confidence": 0.95,
                "status": "verified" if ds_status_str.upper() == "VERIFIED" else "inferred",
            })
            p1_difficulty = "Low"
        else:
            p1_tasks.append(
                RoadmapTask(
                    phase=1,
                    task="Identify public proxy benchmark dataset (flagged: no named benchmark verified in published metadata)",
                    description="No public benchmark dataset verified in published metadata; acquire domain standard proxy.",
                    dependency="None",
                    input="Public proxy archives",
                    output="Proxy dataset files",
                    technology="public open-data portal",
                    difficulty="Medium",
                    evidence_status="UNKNOWN",
                    evidence_source="Publisher metadata absent",
                    supporting_papers=[primary_paper_name],
                )
            )
            p1_tasks.append(
                RoadmapTask(
                    phase=1,
                    task="Construct synthetic validation samples to bootstrap data pipeline",
                    description="Generate mock/synthetic validation samples to unblock downstream engineering steps.",
                    dependency="Identify public proxy benchmark dataset",
                    input="Data schema specifications",
                    output="Synthetic data validation suite",
                    technology="numpy / faker",
                    difficulty="Medium",
                    evidence_status="UNKNOWN",
                    evidence_source="Generic Scaffold",
                    supporting_papers=[primary_paper_name],
                )
            )
            p1_evidence.append({
                "component": "Dataset",
                "value": "Unknown / Domain Benchmark Proxy",
                "source_paper": primary_paper_name,
                "evidence": "No named dataset confirmed in published paper metadata. Standard domain proxy required.",
                "confidence": 0.30,
                "status": "unknown",
            })
            p1_difficulty = "Medium"

        if public_datasets_only:
            p1_tasks.append(
                RoadmapTask(
                    phase=1,
                    task="Enforce constraint: Audit dataset licensing to ensure strict public/open-access availability",
                    description="Verify permissive open-access data licensing across archives (HuggingFace / Zenodo / Kaggle).",
                    dependency="None",
                    input="Dataset licenses",
                    output="License verification compliance report",
                    technology="license auditor",
                    difficulty="Low",
                    evidence_status="INFERRED",
                    evidence_source="Constraint: public_datasets_only",
                    supporting_papers=[primary_paper_name],
                )
            )

        p1_phase = RoadmapPhase(
            phase=1,
            title="Phase 1: Research Specification & Benchmark Dataset Acquisition",
            description=f"Synthesize empirical findings from selected research, extract target performance thresholds, and acquire benchmark dataset '{ds_val}'.",
            tasks=[t.task for t in p1_tasks],
            task_items=p1_tasks,
            dependencies=[],
            inputs=["Selected research paper(s)", "Published methodology & equations", "Benchmark archive sources"],
            outputs=["Verified benchmark dataset partitions", "Target metric acceptance threshold specification", "Data acquisition scripts"],
            estimated_difficulty=p1_difficulty,
            evidence=p1_evidence,
            status="planned",
        )

        # =================================================================
        # PHASE 2 — Computational Runtime & Software Stack Setup
        # =================================================================
        p2_tasks: List[RoadmapTask] = []
        p2_evidence: List[Dict[str, Any]] = []

        dep_names = [d.name if hasattr(d, "name") else d.get("name", "") for d in arch_dependencies]
        dep_str = ", ".join(dep_names[:4]) if dep_names else "torch, numpy, fastapi"
        hw_spec = arch_hardware[0].requirement if arch_hardware and hasattr(arch_hardware[0], "requirement") else (arch_hardware[0].get("requirement") if arch_hardware else "GPU Accelerator (>= 16GB VRAM)")

        p2_tasks.append(
            RoadmapTask(
                phase=2,
                task="Initialize isolated Python 3.10+ virtual environment and dependency lockfile",
                description="Set up isolated virtualenv with pinned dependency locks.",
                dependency="Phase 1: Research Specification & Benchmark Dataset Acquisition",
                input="Python 3.10+ runtime",
                output="Virtualenv workspace and requirements.lock",
                technology="python venv / poetry",
                difficulty="Low",
                evidence_status="VERIFIED",
                evidence_source="Standard Engineering Protocol",
                supporting_papers=[primary_paper_name],
            )
        )
        p2_tasks.append(
            RoadmapTask(
                phase=2,
                task=f"Install deep learning frameworks and libraries ({dep_str})",
                description=f"Install and verify accelerator bindings for: {dep_str}.",
                dependency="Initialize isolated Python 3.10+ virtual environment and dependency lockfile",
                input="Dependency package list",
                output="Installed, verified dependency environment",
                technology="pip / conda",
                difficulty="Low",
                evidence_status="VERIFIED",
                evidence_source="Architecture Dependencies",
                supporting_papers=[primary_paper_name],
            )
        )

        if verified_code_urls:
            code_url = verified_code_urls[0]
            p2_tasks.append(
                RoadmapTask(
                    phase=2,
                    task=f"Clone and verify official paper repository: {code_url}",
                    description=f"Clone official paper repository '{code_url}', verify commits, and inspect requirements for module dependencies ({comp_model.value if comp_model else 'model'}).",
                    dependency="Install deep learning frameworks and libraries",
                    input="Official Git repository URL",
                    output="Cloned, verified repository source tree",
                    technology="git / python",
                    difficulty="Low",
                    evidence_status="VERIFIED",
                    evidence_source=code_url,
                    supporting_papers=[primary_paper_name],
                )
            )
            p2_evidence.append({
                "component": "Repository",
                "value": code_url,
                "source_paper": primary_paper_name,
                "evidence": f"Verified implementation repository: {code_url}",
                "confidence": 0.98,
                "status": "verified",
            })
            p2_difficulty = "Low"
        else:
            p2_tasks.append(
                RoadmapTask(
                    phase=2,
                    task=f"Bootstrap clean modular repository scaffolding for '{primary_paper_name}'",
                    description="Repository URL not confirmed in published paper metadata; initialize clean modular package scaffolding.",
                    dependency="Install deep learning frameworks and libraries",
                    input="Clean repository workspace",
                    output="Modular src/ package structure with test harness",
                    technology="python / pytest",
                    difficulty="Medium",
                    evidence_status="UNKNOWN",
                    evidence_source="Paper metadata: code_url absent",
                    supporting_papers=[primary_paper_name],
                )
            )
            p2_evidence.append({
                "component": "Repository",
                "value": "Clean Repository Scaffolding",
                "source_paper": primary_paper_name,
                "evidence": "No public repository URL identified in paper metadata. Clean package scaffolding required.",
                "confidence": 0.40,
                "status": "unknown",
            })
            p2_difficulty = "Medium"

        # Hardware handling
        if has_verified_hw and hw_ev_str:
            p2_tasks.append(
                RoadmapTask(
                    phase=2,
                    task=f"Configure compute accelerators per paper specifications: {hw_ev_str}",
                    description=f"Allocate accelerator node matching paper specification: {hw_ev_str}.",
                    dependency="None",
                    input="Hardware node",
                    output="Configured accelerator environment",
                    technology="CUDA / NVIDIA Drivers",
                    difficulty="Medium",
                    evidence_status="VERIFIED",
                    evidence_source=hw_ev_str,
                    supporting_papers=[primary_paper_name],
                )
            )
            p2_evidence.append({
                "component": "Hardware",
                "value": hw_ev_str,
                "source_paper": primary_paper_name,
                "evidence": hw_ev_str,
                "confidence": 0.88,
                "status": "verified",
            })
        else:
            p2_tasks.append(
                RoadmapTask(
                    phase=2,
                    task=f"Hardware profile: UNKNOWN in published paper metadata. Allocate standard baseline accelerator ({hw_spec}) and profile during execution",
                    description="Hardware specification not detailed in publication; allocate standard GPU and profile during execution.",
                    dependency="None",
                    input="Compute node",
                    output="Operational baseline accelerator",
                    technology="CUDA / PyTorch",
                    difficulty="Low",
                    evidence_status="UNKNOWN",
                    evidence_source="Paper metadata: hardware absent",
                    supporting_papers=[primary_paper_name],
                )
            )
            p2_evidence.append({
                "component": "Hardware",
                "value": "Unknown / Baseline Accelerator Profile",
                "source_paper": primary_paper_name,
                "evidence": "Hardware configuration not explicitly specified in published paper metadata.",
                "confidence": 0.35,
                "status": "unknown",
            })

        if is_limited_gpu:
            p2_tasks.append(
                RoadmapTask(
                    phase=2,
                    task=f"Enforce constraint: Install quantization packages (bitsandbytes, accelerate) to operate within <= {max_vram_gb:.0f}GB VRAM",
                    description=f"Install memory-reduction libraries to guarantee peak consumption remains under <= {max_vram_gb:.0f}GB VRAM.",
                    dependency="Install deep learning frameworks and libraries",
                    input="Virtualenv",
                    output="Installed bitsandbytes and accelerate libraries",
                    technology="bitsandbytes / accelerate",
                    difficulty="Low",
                    evidence_status="INFERRED",
                    evidence_source=f"Constraint: max_gpu_memory_gb={max_vram_gb}",
                    supporting_papers=[primary_paper_name],
                )
            )
        if open_source_only:
            p2_tasks.append(
                RoadmapTask(
                    phase=2,
                    task="Enforce constraint: Audit all third-party dependencies to verify permissive open-source licenses (MIT, Apache-2.0, BSD)",
                    description="Scan virtual environment to verify zero proprietary or incompatible commercial dependencies.",
                    dependency="Install deep learning frameworks and libraries",
                    input="Installed package metadata",
                    output="Open-source license compliance report",
                    technology="pip-licenses",
                    difficulty="Low",
                    evidence_status="INFERRED",
                    evidence_source="Constraint: open_source_only",
                    supporting_papers=[primary_paper_name],
                )
            )

        p2_phase = RoadmapPhase(
            phase=2,
            title="Phase 2: Development Environment & Software Stack Setup",
            description=f"Provision computational runtime, configure acceleration ({hw_spec}), install dependencies ({dep_str}), and integrate verified code repositories.",
            tasks=[t.task for t in p2_tasks],
            task_items=p2_tasks,
            dependencies=["Phase 1: Research Specification & Benchmark Dataset Acquisition"],
            inputs=["Hardware accelerator node", "Python 3.10+ runtime", "Package package managers (pip/poetry)"],
            outputs=["Reproducible virtual environment", "Pinned dependency lockfile", "Repository workspace setup"],
            estimated_difficulty=p2_difficulty,
            evidence=p2_evidence,
            status="planned",
        )

        # =================================================================
        # PHASE 3 — Data Preparation & Preprocessing Pipeline
        # =================================================================
        p3_tasks: List[RoadmapTask] = []
        p3_evidence: List[Dict[str, Any]] = []

        preproc_val = comp_preproc.value if comp_preproc else "Standard Data Cleaning & Normalization"
        preproc_status = comp_preproc.evidence_status.value if comp_preproc else "INFERRED"
        preproc_tech = ", ".join(comp_preproc.technologies) if comp_preproc and comp_preproc.technologies else "PyTorch / torchvision / scikit-learn"

        p3_tasks.append(
            RoadmapTask(
                phase=3,
                task=f"Construct automated ingestion loader for raw benchmark datasets ({ds_val})",
                description=f"Build automated dataset parser ingesting raw files from Phase 1 into memory-efficient stream generators for '{ds_val}'.",
                dependency="Phase 1: Research Specification & Benchmark Dataset Acquisition",
                input=f"Raw dataset files ({ds_val})",
                output="Data loading generator / PyTorch Dataset class",
                technology="PyTorch DataLoader / polars / pandas",
                difficulty="Medium",
                evidence_status=ds_status_str.upper(),
                evidence_source=comp_dataset.evidence_source if comp_dataset else None,
                supporting_papers=[primary_paper_name],
            )
        )
        p3_tasks.append(
            RoadmapTask(
                phase=3,
                task="Partition data into deterministic 80/10/10 train/validation/test splits with stratified sampling",
                description="Split dataset deterministically into 80% train, 10% validation, and 10% test splits.",
                dependency=f"Construct automated ingestion loader for raw benchmark datasets ({ds_val})",
                input="Ingested dataset partitions",
                output="Deterministic split indices",
                technology="scikit-learn train_test_split",
                difficulty="Low",
                evidence_status="VERIFIED",
                evidence_source="Standard Machine Learning Protocol",
                supporting_papers=[primary_paper_name],
            )
        )

        # Domain/Architecture-specific preprocessing tasks
        preproc_lower = preproc_val.lower()
        if "skull" in preproc_lower or "strip" in preproc_lower:
            p3_tasks.append(
                RoadmapTask(
                    phase=3,
                    task="Implement skull stripping and brain volume isolation",
                    description="Segment cranial vault and isolate intracranial brain parenchyma from volumetric MRI scans.",
                    dependency=f"Construct automated ingestion loader for raw benchmark datasets ({ds_val})",
                    input=f"Raw volumetric MRI scans ({ds_val})",
                    output="Skull-stripped 3D brain masks and intracranial volumes",
                    technology="ANTs / BET / PyTorch",
                    difficulty="Medium",
                    evidence_status=preproc_status,
                    evidence_source=comp_preproc.evidence_source if comp_preproc else None,
                    supporting_papers=[primary_paper_name],
                )
            )
        if "intensity" in preproc_lower or "normal" in preproc_lower:
            p3_tasks.append(
                RoadmapTask(
                    phase=3,
                    task="Implement intensity normalization and z-score scaling",
                    description="Standardize voxel/pixel intensity distributions across imaging modalities to zero mean and unit variance.",
                    dependency="Implement skull stripping and brain volume isolation" if ("skull" in preproc_lower) else f"Construct automated ingestion loader for raw benchmark datasets ({ds_val})",
                    input="Unnormalized raw or stripped scan slices",
                    output="Intensity-normalized and z-score standardized image arrays",
                    technology="TorchIO / numpy / scikit-image",
                    difficulty="Low",
                    evidence_status=preproc_status,
                    evidence_source=comp_preproc.evidence_source if comp_preproc else None,
                    supporting_papers=[primary_paper_name],
                )
            )
        if "resiz" in preproc_lower or "224" in preproc_lower:
            p3_tasks.append(
                RoadmapTask(
                    phase=3,
                    task="Implement image resizing and spatial dimension standardization",
                    description="Standardize input image dimensions to uniform spatial resolution suitable for backbone feature extraction.",
                    dependency=f"Construct automated ingestion loader for raw benchmark datasets ({ds_val})",
                    input="Raw variable-resolution images",
                    output="Standardized spatial resolution image tensors (224x224)",
                    technology="torchvision.transforms / albumentations",
                    difficulty="Low",
                    evidence_status=preproc_status,
                    evidence_source=comp_preproc.evidence_source if comp_preproc else None,
                    supporting_papers=[primary_paper_name],
                )
            )
        if "augment" in preproc_lower or "jitter" in preproc_lower or "rotation" in preproc_lower:
            p3_tasks.append(
                RoadmapTask(
                    phase=3,
                    task="Implement data augmentation pipeline (geometric transforms, color jitter)",
                    description="Apply stochastic geometric rotations, horizontal flips, and color perturbations to prevent overfitting on foliar and visual datasets.",
                    dependency="Implement image resizing and spatial dimension standardization" if ("resiz" in preproc_lower) else f"Construct automated ingestion loader for raw benchmark datasets ({ds_val})",
                    input="Standardized image batches",
                    output="Stochastically augmented training tensors",
                    technology="albumentations / torchvision",
                    difficulty="Low",
                    evidence_status=preproc_status,
                    evidence_source=comp_preproc.evidence_source if comp_preproc else None,
                    supporting_papers=[primary_paper_name],
                )
            )
        if "window" in preproc_lower or "aggregat" in preproc_lower:
            p3_tasks.append(
                RoadmapTask(
                    phase=3,
                    task="Implement sliding time-window feature aggregation",
                    description="Aggregate multi-step temporal transactions into fixed observation windows for sequential representation.",
                    dependency=f"Construct automated ingestion loader for raw benchmark datasets ({ds_val})",
                    input="Raw temporal event and transaction logs",
                    output="Time-window aggregated feature matrices",
                    technology="polars / pandas",
                    difficulty="Medium",
                    evidence_status=preproc_status,
                    evidence_source=comp_preproc.evidence_source if comp_preproc else None,
                    supporting_papers=[primary_paper_name],
                )
            )
        if "smote" in preproc_lower or "oversampling" in preproc_lower or "imbalanc" in preproc_lower:
            p3_tasks.append(
                RoadmapTask(
                    phase=3,
                    task="Implement SMOTE oversampling for extreme class imbalance",
                    description="Synthesize minority class feature vectors to balance training distribution across anomalous and fraud classes.",
                    dependency="Implement sliding time-window feature aggregation" if ("window" in preproc_lower) else f"Construct automated ingestion loader for raw benchmark datasets ({ds_val})",
                    input="Imbalanced training feature matrices",
                    output="Synthetically balanced class distribution tensors",
                    technology="imbalanced-learn",
                    difficulty="Medium",
                    evidence_status=preproc_status,
                    evidence_source=comp_preproc.evidence_source if comp_preproc else None,
                    supporting_papers=[primary_paper_name],
                )
            )

        p3_tasks.append(
            RoadmapTask(
                phase=3,
                task=f"Implement modality-specific feature normalization and transformation: {preproc_val}",
                description=f"Execute paper-specified transformations: {preproc_val}. Evidence: {comp_preproc.evidence if comp_preproc else 'Standard transformation'}.",
                dependency="Partition data into deterministic 80/10/10 train/validation/test splits with stratified sampling",
                input="Raw data records / images",
                output="Cleaned, standardized, model-ready tensor inputs",
                technology=preproc_tech,
                difficulty="Medium",
                evidence_status=preproc_status,
                evidence_source=comp_preproc.evidence_source if comp_preproc else None,
                supporting_papers=[primary_paper_name],
            )
        )
        p3_tasks.append(
            RoadmapTask(
                phase=3,
                task="Execute data integrity unit tests asserting zero leakage between train and test splits",
                description="Run automated unit tests asserting zero index overlap and consistent distribution across partitions.",
                dependency=f"Implement modality-specific feature normalization and transformation: {preproc_val}",
                input="Transformed dataset partitions",
                output="Data integrity unit test report",
                technology="pytest",
                difficulty="Low",
                evidence_status="INFERRED",
                evidence_source="ResearchMind Quality Verification Protocol",
                supporting_papers=[primary_paper_name],
            )
        )

        if comp_preproc:
            p3_evidence.append({
                "component": "Preprocessing",
                "value": preproc_val,
                "source_paper": comp_preproc.source_paper,
                "evidence": comp_preproc.evidence,
                "status": comp_preproc.status.value.lower(),
            })

        p3_phase = RoadmapPhase(
            phase=3,
            title="Phase 3: Data Ingestion, Cleaning & Preprocessing",
            description=f"Construct automated data loading pipelines, apply preprocessing transformations '{preproc_val}', and generate standardized training/validation splits.",
            tasks=[t.task for t in p3_tasks],
            task_items=p3_tasks,
            dependencies=[
                "Phase 1: Research Specification & Benchmark Dataset Acquisition",
                "Phase 2: Development Environment & Software Stack Setup",
            ],
            inputs=["Raw downloaded datasets from Phase 1", "Preprocessing pipeline specifications"],
            outputs=["Cleaned, tokenized/normalized tensors", "Deterministic train/val/test split indices", "Data integrity unit test suite"],
            estimated_difficulty="Medium",
            evidence=p3_evidence or p1_evidence[:1],
            status="planned",
        )

        # =================================================================
        # PHASE 4 — Model Architecture & Backbone Implementation
        # =================================================================
        p4_tasks: List[RoadmapTask] = []
        p4_evidence: List[Dict[str, Any]] = []

        # Feature Extraction decomposition
        feat_val = comp_feat.value if comp_feat else None
        feat_status = comp_feat.evidence_status.value if comp_feat else "INFERRED"
        if feat_val and "unknown" not in feat_val.lower():
            if "resnet" in feat_val.lower():
                p4_tasks.append(
                    RoadmapTask(
                        phase=4,
                        task="Extract ResNet50 deep residual features",
                        description="Instantiate pre-trained ResNet50 backbone to extract dense visual representation vectors.",
                        dependency="Phase 3: Data Ingestion, Cleaning & Preprocessing",
                        input="Preprocessed normalized tensors from Phase 3",
                        output="Deep residual feature embedding tensors (2048-dim)",
                        technology="torchvision.models.resnet50",
                        difficulty="Medium",
                        evidence_status=feat_status,
                        evidence_source=comp_feat.evidence_source if comp_feat else None,
                        supporting_papers=[primary_paper_name],
                    )
                )
            elif "efficientnet" in feat_val.lower():
                p4_tasks.append(
                    RoadmapTask(
                        phase=4,
                        task="Initialize EfficientNet-B0 feature extractor",
                        description="Configure EfficientNet-B0 depthwise separable convolutional backbone for spatial feature extraction.",
                        dependency="Phase 3: Data Ingestion, Cleaning & Preprocessing",
                        input="Standardized image batches from Phase 3",
                        output="Depthwise separable feature representations",
                        technology="timm / torchvision",
                        difficulty="Medium",
                        evidence_status=feat_status,
                        evidence_source=comp_feat.evidence_source if comp_feat else None,
                        supporting_papers=[primary_paper_name],
                    )
                )
            elif "temporal" in feat_val.lower() or "attention" in feat_val.lower():
                p4_tasks.append(
                    RoadmapTask(
                        phase=4,
                        task="Implement Multi-Head Self-Attention Temporal Window Encoder",
                        description="Construct multi-head self-attention layers to encode sequential event dependencies across transaction windows.",
                        dependency="Phase 3: Data Ingestion, Cleaning & Preprocessing",
                        input="Time-window feature vectors from Phase 3",
                        output="Attention-weighted contextual latent representations",
                        technology="torch.nn.MultiheadAttention",
                        difficulty="High",
                        evidence_status=feat_status,
                        evidence_source=comp_feat.evidence_source if comp_feat else None,
                        supporting_papers=[primary_paper_name],
                    )
                )
            else:
                p4_tasks.append(
                    RoadmapTask(
                        phase=4,
                        task=f"Implement feature extraction pipeline: {feat_val}",
                        description=f"Extract high-dimensional representation embeddings using {feat_val}.",
                        dependency="Phase 3: Data Ingestion, Cleaning & Preprocessing",
                        input="Preprocessed tensors from Phase 3",
                        output="Extracted feature embedding vectors",
                        technology=", ".join(comp_feat.technologies) if (comp_feat and comp_feat.technologies) else "PyTorch",
                        difficulty="Medium",
                        evidence_status=feat_status,
                        evidence_source=comp_feat.evidence_source if comp_feat else None,
                        supporting_papers=[primary_paper_name],
                    )
                )

        model_val = comp_model.value if comp_model else "Neural Network Backbone"
        model_status = comp_model.evidence_status.value if comp_model else "INFERRED"
        model_tech = ", ".join(comp_model.technologies) if comp_model and comp_model.technologies else "PyTorch / torchvision / transformers"

        # Model component classification
        if "svm" in model_val.lower():
            p4_tasks.append(
                RoadmapTask(
                    phase=4,
                    task="Construct Support Vector Machine (SVM) Decision Boundary Classifier",
                    description="Formulate kernelized support vector machine to maximize separation margin on extracted deep features.",
                    dependency="Extract ResNet50 deep residual features" if (feat_val and "resnet" in feat_val.lower()) else "Phase 2: Development Environment & Software Stack Setup",
                    input="Extracted feature vectors",
                    output="Configured SVM decision boundary classifier instance",
                    technology="scikit-learn SVC",
                    difficulty="Medium",
                    evidence_status=model_status,
                    evidence_source=comp_model.evidence_source if comp_model else None,
                    supporting_papers=[primary_paper_name],
                )
            )
            p4_difficulty = "Medium"
        elif "classifier" in model_val.lower() or "softmax" in model_val.lower():
            p4_tasks.append(
                RoadmapTask(
                    phase=4,
                    task=f"Construct {model_val}",
                    description=f"Implement classification head with linear projection and activation: {model_val}.",
                    dependency="Initialize EfficientNet-B0 feature extractor" if (feat_val and "efficientnet" in feat_val.lower()) else "Phase 2: Development Environment & Software Stack Setup",
                    input="Feature extractor layer outputs",
                    output="End-to-end model classification graph",
                    technology="PyTorch nn.Linear / softmax",
                    difficulty="Medium",
                    evidence_status=model_status,
                    evidence_source=comp_model.evidence_source if comp_model else None,
                    supporting_papers=[primary_paper_name],
                )
            )
            p4_difficulty = "Medium"
        elif verified_code_urls:
            p4_tasks.append(
                RoadmapTask(
                    phase=4,
                    task=f"Import modular '{model_val}' definitions directly from verified repository",
                    description=f"Import neural layer classes for '{model_val}' directly from the verified cloned paper repository.",
                    dependency="Phase 2: Development Environment & Software Stack Setup",
                    input="Cloned official paper repository from Phase 2",
                    output=f"Imported, verified PyTorch model class for {model_val}",
                    technology="python / PyTorch",
                    difficulty="Medium",
                    evidence_status="VERIFIED",
                    evidence_source=verified_code_urls[0],
                    supporting_papers=[primary_paper_name],
                )
            )
            p4_difficulty = "Medium"
        else:
            p4_tasks.append(
                RoadmapTask(
                    phase=4,
                    task=f"Implement '{model_val}' architecture and mathematical layers from paper description",
                    description=f"Construct model architecture '{model_val}' and custom mathematical layers from paper descriptions. Evidence: {comp_model.evidence if comp_model else 'Architecture description'}.",
                    dependency="Phase 2: Development Environment & Software Stack Setup",
                    input="Model architecture equations & specifications",
                    output=f"Custom PyTorch model class for {model_val}",
                    technology=model_tech,
                    difficulty="High",
                    evidence_status=model_status,
                    evidence_source=comp_model.evidence_source if comp_model else None,
                    supporting_papers=[primary_paper_name],
                )
            )
            p4_difficulty = "High"

        p4_tasks.append(
            RoadmapTask(
                phase=4,
                task=f"Verify layer dimensionalities, weight parameter counts, and tensor propagation for {model_val}",
                description="Assert forward-pass input-to-output shape compatibility and total parameter footprint against paper.",
                dependency=f"Verify layer propagation for {model_val}",
                input=f"Instantiated {model_val} model instance",
                output="Parameter footprint audit report",
                technology="torchinfo / pytest",
                difficulty="Low",
                evidence_status="VERIFIED" if model_status == "VERIFIED" else "INFERRED",
                evidence_source="Paper Architecture Spec",
                supporting_papers=[primary_paper_name],
            )
        )
        p4_tasks.append(
            RoadmapTask(
                phase=4,
                task=f"Validate layer equations, activation functions, and normalization blocks for {model_val}",
                description="Verify mathematical correctness of neural activations and normalization against formulas in paper.",
                dependency=f"Verify layer dimensionalities, weight parameter counts, and tensor propagation for {model_val}",
                input=f"Implemented {model_val} model class",
                output="Mathematical layer validation test suite",
                technology="pytest / torch",
                difficulty="Medium",
                evidence_status=model_status,
                evidence_source=comp_model.evidence_source if comp_model else None,
                supporting_papers=[primary_paper_name],
            )
        )

        p4_evidence.append({
            "component": "Model",
            "value": model_val,
            "source_paper": primary_paper_name,
            "evidence": f"Model architecture identified in research: {model_val}",
            "confidence": 0.90 if model_status == "VERIFIED" else 0.50,
            "status": "verified" if model_status == "VERIFIED" else "inferred",
        })

        if is_limited_gpu:
            p4_tasks.append(
                RoadmapTask(
                    phase=4,
                    task=f"Enforce constraint: Wrap model with Parameter-Efficient Fine-Tuning (PEFT / LoRA) or 8-bit quantization for <= {max_vram_gb:.0f}GB VRAM",
                    description="Wrap backbone with LoRA / PEFT adapters to reduce VRAM consumption during fine-tuning.",
                    dependency=f"Implement '{model_val}' architecture" if not verified_code_urls else f"Import modular '{model_val}' definitions",
                    input=f"Instantiated {model_val}",
                    output="Memory-optimized PEFT model instance",
                    technology="peft / bitsandbytes",
                    difficulty="Medium",
                    evidence_status="INFERRED",
                    evidence_source=f"Constraint: max_gpu_memory_gb={max_vram_gb}",
                    supporting_papers=[primary_paper_name],
                )
            )
        if is_limited_time:
            p4_tasks.append(
                RoadmapTask(
                    phase=4,
                    task="Enforce constraint: Instantiate pre-trained checkpoint weights from PyTorch Hub / HuggingFace; skip training from scratch",
                    description="Load pretrained weights from hub to fast-track implementation within tight time budget.",
                    dependency=f"Implement '{model_val}' architecture" if not verified_code_urls else f"Import modular '{model_val}' definitions",
                    input="Pretrained weight hub identifier",
                    output="Pretrained model instance ready for transfer learning",
                    technology="torchvision / transformers / huggingface_hub",
                    difficulty="Low",
                    evidence_status="INFERRED",
                    evidence_source="Constraint: limited_time",
                    supporting_papers=[primary_paper_name],
                )
            )

        p4_phase = RoadmapPhase(
            phase=4,
            title="Phase 4: Model Architecture & Neural Backbone Implementation",
            description=f"Implement core '{model_val}' architecture, configure forward pass operations, and verify parameter footprint.",
            tasks=[t.task for t in p4_tasks],
            task_items=p4_tasks,
            dependencies=["Phase 2: Development Environment & Software Stack Setup"],
            inputs=["Model architecture specifications", "Initialized development runtime from Phase 2"],
            outputs=["PyTorch / ONNX model class implementation", "Parameter count and footprint report", "Forward-pass sanity test script"],
            estimated_difficulty=p4_difficulty,
            evidence=p4_evidence,
            status="planned",
        )

        # =================================================================
        # PHASE 5 — Classifier Head & Model Training Formulation
        # =================================================================
        p5_tasks: List[RoadmapTask] = []
        p5_evidence: List[Dict[str, Any]] = []

        train_val = comp_train.value if comp_train else "AdamW Optimizer with Cosine Annealing Learning Rate"
        train_status = comp_train.evidence_status.value if comp_train else "INFERRED"

        if "svm" in model_val.lower():
            p5_tasks.append(
                RoadmapTask(
                    phase=5,
                    task="Train SVM classifier on extracted feature embeddings",
                    description="Fit support vector classifier with optimal hyperparameter search (C, gamma) on extracted deep features.",
                    dependency="Construct Support Vector Machine (SVM) Decision Boundary Classifier",
                    input="Extracted training features and ground truth labels",
                    output="Trained SVM model artifact",
                    technology="scikit-learn / libsvm",
                    difficulty="Medium",
                    evidence_status=train_status,
                    evidence_source=comp_train.evidence_source if comp_train else None,
                    supporting_papers=[primary_paper_name],
                )
            )
        else:
            p5_tasks.append(
                RoadmapTask(
                    phase=5,
                    task=f"Configure training loss: {train_val}",
                    description=f"Formulate loss function and convergence objectives: {train_val}. Evidence: {comp_train.evidence if comp_train else 'Standard training'}.",
                    dependency="Phase 4: Model Architecture & Neural Backbone Implementation",
                    input="Instantiated model instance and preprocessed batches",
                    output="Configured loss objective function",
                    technology="PyTorch loss functions",
                    difficulty="Medium",
                    evidence_status=train_status,
                    evidence_source=comp_train.evidence_source if comp_train else None,
                    supporting_papers=[primary_paper_name],
                )
            )

        p5_tasks.append(
            RoadmapTask(
                phase=5,
                task=f"Set up optimizer and learning rate scheduler: {train_val}",
                description="Configure learning rate scheduling with warm-up steps and weight decay.",
                dependency=p5_tasks[-1].task,
                input="Model parameters",
                output="Configured optimizer and scheduler",
                technology="torch.optim",
                difficulty="Low",
                evidence_status=train_status,
                evidence_source=comp_train.evidence_source if comp_train else None,
                supporting_papers=[primary_paper_name],
            )
        )

        if "fine-tune" in primary_doc.title.lower() or "fine-tuning" in train_val.lower() or "efficientnet" in model_val.lower():
            p5_tasks.append(
                RoadmapTask(
                    phase=5,
                    task=f"Fine-tune classifier and transfer learning layers: {model_val}",
                    description=f"Unfreeze top convolutional layers and fine-tune classifier weights with differential learning rates.",
                    dependency=f"Set up optimizer and learning rate scheduler: {train_val}",
                    input="Pretrained backbone weights",
                    output="Fine-tuned weights on target dataset",
                    technology="PyTorch / timm",
                    difficulty="Medium",
                    evidence_status=train_status,
                    evidence_source=comp_train.evidence_source if comp_train else None,
                    supporting_papers=[primary_paper_name],
                )
            )

        p5_tasks.append(
            RoadmapTask(
                phase=5,
                task="Implement automated checkpointing callback saving top-3 checkpoints by validation loss",
                description="Persist model weights (.pt / .safetensors) whenever validation metric improves.",
                dependency=f"Set up optimizer and learning rate scheduler: {train_val}",
                input="Training loop validation hook",
                output="Serialized model checkpoint archive",
                technology="PyTorch checkpointing",
                difficulty="Low",
                evidence_status="INFERRED",
                evidence_source="Standard Training Engineering",
                supporting_papers=[primary_paper_name],
            )
        )
        p5_tasks.append(
            RoadmapTask(
                phase=5,
                task=f"Execute training loop with gradient norm clipping and step logging for {model_val}",
                description=f"Run multi-epoch training loop with gradient norm clipping and real-time loss tracking for {model_val}.",
                dependency="Implement automated checkpointing callback saving top-3 checkpoints by validation loss",
                input="Training data batches from Phase 3",
                output="Trained model weights and loss curves",
                technology="PyTorch / TensorBoard",
                difficulty="High" if not is_limited_time else "Medium",
                evidence_status="INFERRED",
                evidence_source="ResearchMind Training Execution",
                supporting_papers=[primary_paper_name],
            )
        )

        if is_limited_gpu:
            p5_tasks.append(
                RoadmapTask(
                    phase=5,
                    task=f"Enforce constraint: Train with micro-batch size (1-2), gradient accumulation = 16, and FP16/BF16 mixed precision for <= {max_vram_gb:.0f}GB VRAM",
                    description="Optimize memory footprint using gradient accumulation and half precision.",
                    dependency="Execute training loop",
                    input="GPU device",
                    output="Memory-optimized training execution",
                    technology="torch.cuda.amp",
                    difficulty="Medium",
                    evidence_status="INFERRED",
                    evidence_source=f"Constraint: max_gpu_memory_gb={max_vram_gb}",
                    supporting_papers=[primary_paper_name],
                )
            )
        if is_limited_time:
            p5_tasks.append(
                RoadmapTask(
                    phase=5,
                    task="Enforce constraint: Execute fast-track 1-3 epoch fine-tuning run with early stopping",
                    description="Execute fast-track fine-tuning loop completing within 1-week budget.",
                    dependency="Execute training loop",
                    input="Pretrained checkpoint",
                    output="Fine-tuned model checkpoint",
                    technology="early stopping callback",
                    difficulty="Low",
                    evidence_status="INFERRED",
                    evidence_source="Constraint: limited_time",
                    supporting_papers=[primary_paper_name],
                )
            )

        if comp_train:
            p5_evidence.append({
                "component": "Training",
                "value": train_val,
                "source_paper": comp_train.source_paper,
                "evidence": comp_train.evidence,
                "status": comp_train.status.value.lower(),
            })

        p5_phase = RoadmapPhase(
            phase=5,
            title="Phase 5: Model Training, Loss Function Formulation & Checkpointing",
            description=f"Execute training loop, configure convergence optimization ({train_val}), log loss trajectories, and persist serialized checkpoints.",
            tasks=[t.task for t in p5_tasks],
            task_items=p5_tasks,
            dependencies=[
                "Phase 3: Data Ingestion, Cleaning & Preprocessing",
                "Phase 4: Model Architecture & Neural Backbone Implementation",
            ],
            inputs=["Preprocessed data loaders from Phase 3", "Model backbone from Phase 4", "Hyperparameter configuration"],
            outputs=["Trained model weights (.pt / .safetensors)", "Training and validation loss curves", "Convergence telemetry logs"],
            estimated_difficulty="Medium" if is_limited_time else "High",
            evidence=p5_evidence or p4_evidence[:1],
            status="planned",
        )

        # =================================================================
        # PHASE 6 — Empirical Evaluation & Benchmark Validation
        # =================================================================
        p6_tasks: List[RoadmapTask] = []
        p6_evidence: List[Dict[str, Any]] = []

        eval_val = comp_eval.value if comp_eval else "Standard Benchmark Validation Metrics"
        eval_status = comp_eval.evidence_status.value if comp_eval else "INFERRED"

        p6_tasks.append(
            RoadmapTask(
                phase=6,
                task=f"Evaluate trained checkpoint on holdout test partition using paper benchmark metrics ({eval_val})",
                description=f"Assess model generalization on holdout partitions using '{eval_val}'. Evidence: {comp_eval.evidence if comp_eval else 'Published metrics'}.",
                dependency="Phase 5: Model Training, Loss Function Formulation & Checkpointing",
                input="Best checkpoint from Phase 5 and holdout test split from Phase 3",
                output=f"Comprehensive metric scorecard ({eval_val})",
                technology="scikit-learn / torchmetrics",
                difficulty="Medium",
                evidence_status=eval_status,
                evidence_source=comp_eval.evidence_source if comp_eval else None,
                supporting_papers=[primary_paper_name],
            )
        )

        eval_lower = eval_val.lower()
        if "sensitivity" in eval_lower or "specificity" in eval_lower or "auc" in eval_lower:
            p6_tasks.append(
                RoadmapTask(
                    phase=6,
                    task="Evaluate sensitivity, specificity, and AUC-ROC on holdout test partition",
                    description="Calculate true positive rate (sensitivity), true negative rate (specificity), and area under ROC curve for clinical discrimination.",
                    dependency=f"Evaluate trained checkpoint on holdout test partition using paper benchmark metrics ({eval_val})",
                    input="Model predictions and ground truth labels",
                    output="Clinical validation metrics scorecard (Sensitivity, Specificity, AUC)",
                    technology="scikit-learn / torchmetrics",
                    difficulty="Medium",
                    evidence_status=eval_status,
                    evidence_source=comp_eval.evidence_source if comp_eval else None,
                    supporting_papers=[primary_paper_name],
                )
            )
        elif "accuracy" in eval_lower or "macro-f1" in eval_lower:
            p6_tasks.append(
                RoadmapTask(
                    phase=6,
                    task="Evaluate Top-1 accuracy and Macro-F1 score across target disease classes",
                    description="Assess per-class precision, recall, macro-averaged F1 score, and overall top-1 accuracy on unseen benchmark test images.",
                    dependency=f"Evaluate trained checkpoint on holdout test partition using paper benchmark metrics ({eval_val})",
                    input="Model test predictions and ground truth labels",
                    output="Per-class precision, recall, and Macro-F1 report",
                    technology="scikit-learn classification_report",
                    difficulty="Medium",
                    evidence_status=eval_status,
                    evidence_source=comp_eval.evidence_source if comp_eval else None,
                    supporting_papers=[primary_paper_name],
                )
            )
        elif "pr-auc" in eval_lower or "detection rate" in eval_lower or "fpr" in eval_lower:
            p6_tasks.append(
                RoadmapTask(
                    phase=6,
                    task="Evaluate PR-AUC and detection rate at 0.1% false-positive rate",
                    description="Compute Precision-Recall curve area and precision at fixed low false alarm rates on highly skewed class distribution.",
                    dependency=f"Evaluate trained checkpoint on holdout test partition using paper benchmark metrics ({eval_val})",
                    input="Model anomaly risk scores and fraud labels",
                    output="PR-AUC and operating point scorecard",
                    technology="scikit-learn precision_recall_curve",
                    difficulty="Medium",
                    evidence_status=eval_status,
                    evidence_source=comp_eval.evidence_source if comp_eval else None,
                    supporting_papers=[primary_paper_name],
                )
            )

        p6_tasks.append(
            RoadmapTask(
                phase=6,
                task="Generate confusion matrix, precision-recall curves, and error analysis across edge cases",
                description="Profile misclassified samples and false-positive distributions across classes.",
                dependency=f"Evaluate trained checkpoint on holdout test partition using paper benchmark metrics ({eval_val})",
                input="Model predictions and ground truth labels",
                output="Confusion matrix and error analysis breakdown report",
                technology="matplotlib / seaborn / scikit-learn",
                difficulty="Low",
                evidence_status="INFERRED",
                evidence_source="ResearchMind Evaluation Protocol",
                supporting_papers=[primary_paper_name],
            )
        )

        # Dynamic Explainability Attribution task
        explain_val = comp_explainability.value if comp_explainability else ""
        if not explain_val:
            combined_context = (target_query + " " + primary_paper_name + " " + model_val + " " + preproc_val).lower()
            if any(k in combined_context for k in ["grad-cam", "cam", "mri", "tumor", "cancer", "crop", "plant", "leaf", "medical"]):
                explain_val = "Grad-CAM Saliency Localization & Visual Attribution"
            elif any(k in combined_context for k in ["shap", "fraud", "transaction", "tabular", "anomaly"]):
                explain_val = "SHAP Tree / Sequence Feature Attribution Explainer"
            else:
                explain_val = "Feature Attribution & Saliency Visualizer"

        if "grad-cam" in explain_val.lower() or "cam" in explain_val.lower():
            p6_tasks.append(
                RoadmapTask(
                    phase=6,
                    task="Implement Grad-CAM saliency localization and visual attribution",
                    description="Compute gradient backpropagation from target class to convolutional activations to generate saliency heatmaps.",
                    dependency=f"Evaluate trained checkpoint on holdout test partition using paper benchmark metrics ({eval_val})",
                    input="Trained model and test samples",
                    output="Grad-CAM heatmaps overlaying input images",
                    technology="captum / pytorch-grad-cam",
                    difficulty="Medium",
                    evidence_status="VERIFIED" if ("grad-cam" in primary_paper_name.lower() or "grad-cam" in (comp_model.evidence if comp_model else "").lower()) else "INFERRED",
                    evidence_source=primary_paper_name,
                    supporting_papers=[primary_paper_name],
                )
            )
        elif "shap" in explain_val.lower():
            p6_tasks.append(
                RoadmapTask(
                    phase=6,
                    task="Implement SHAP feature attribution and sequence risk explanation",
                    description="Calculate Shapley values across sequential features to quantify marginal risk contributions.",
                    dependency=f"Evaluate trained checkpoint on holdout test partition using paper benchmark metrics ({eval_val})",
                    input="Trained model and transaction feature vectors",
                    output="SHAP waterfall and summary attribution plots",
                    technology="shap",
                    difficulty="Medium",
                    evidence_status="VERIFIED" if ("shap" in primary_paper_name.lower()) else "INFERRED",
                    evidence_source=primary_paper_name,
                    supporting_papers=[primary_paper_name],
                )
            )
        else:
            p6_tasks.append(
                RoadmapTask(
                    phase=6,
                    task=f"Implement model interpretability and explainability: {explain_val}",
                    description=f"Formulate feature attribution and saliency maps using {explain_val}.",
                    dependency=f"Evaluate trained checkpoint on holdout test partition using paper benchmark metrics ({eval_val})",
                    input="Trained model and test instances",
                    output="Model interpretability report",
                    technology="captum / scikit-learn",
                    difficulty="Medium",
                    evidence_status="INFERRED",
                    evidence_source=primary_paper_name,
                    supporting_papers=[primary_paper_name],
                )
            )

        p6_tasks.append(
            RoadmapTask(
                phase=6,
                task="Execute component ablation studies to quantify incremental performance lift over reference baseline",
                description="Perform component ablation analysis removing proposed modules to verify incremental lift.",
                dependency=f"Evaluate trained checkpoint on holdout test partition using paper benchmark metrics ({eval_val})",
                input="Model components",
                output="Ablation study comparative summary",
                technology="ablation harness",
                difficulty="Medium",
                evidence_status="INFERRED",
                evidence_source="ResearchMind Evaluation Protocol",
                supporting_papers=[primary_paper_name],
            )
        )
        p6_tasks.append(
            RoadmapTask(
                phase=6,
                task=f"Verify empirical reproduction against published results reported in '{primary_paper_name}'",
                description=f"Assert parity between reproduced scorecard and published numbers reported in '{primary_paper_name}'.",
                dependency=f"Evaluate trained checkpoint on holdout test partition using paper benchmark metrics ({eval_val})",
                input="Empirical scorecard vs published numbers",
                output="Reproduction parity audit scorecard",
                technology="statistical audit",
                difficulty="Low",
                evidence_status="VERIFIED" if not is_scaffold else "UNKNOWN",
                evidence_source=primary_paper_name,
                supporting_papers=[primary_paper_name],
            )
        )

        p6_evidence.append({
            "component": "Evaluation",
            "value": eval_val,
            "source_paper": primary_paper_name,
            "evidence": f"Evaluation benchmark metrics identified in literature: {eval_val}",
            "confidence": 0.92,
            "status": "verified" if eval_status == "VERIFIED" else "inferred",
        })

        p6_phase = RoadmapPhase(
            phase=6,
            title="Phase 6: Empirical Evaluation, Benchmark Validation & Ablations",
            description=f"Assess model generalization on holdout partitions using {eval_val}, perform error profiling, and validate ablation gains.",
            tasks=[t.task for t in p6_tasks],
            task_items=p6_tasks,
            dependencies=["Phase 5: Model Training, Loss Function Formulation & Checkpointing"],
            inputs=["Best trained model checkpoint from Phase 5", "Holdout test split from Phase 3"],
            outputs=["Comprehensive metric scorecard", "Ablation study comparative summary", "Error analysis report and failure modes"],
            estimated_difficulty="Medium",
            evidence=p6_evidence,
            status="planned",
        )

        # =================================================================
        # PHASE 7 — Inference Optimization & Quantization
        # =================================================================
        p7_tasks: List[RoadmapTask] = []
        p7_evidence: List[Dict[str, Any]] = []

        infer_val = comp_infer.value if comp_infer else "Compiled Inference Runtime"
        infer_status = comp_infer.evidence_status.value if comp_infer else "INFERRED"

        p7_tasks.append(
            RoadmapTask(
                phase=7,
                task=f"Export verified PyTorch model checkpoint to ONNX / TorchScript format ({infer_val})",
                description=f"Export PyTorch weights to {infer_val} format with operator fusion. Evidence: {comp_infer.evidence if comp_infer else 'Compilation runtime'}.",
                dependency="Phase 6: Empirical Evaluation, Benchmark Validation & Ablations",
                input="Verified model checkpoint from Phase 6",
                output=f"Compiled inference engine ({infer_val})",
                technology="torch.onnx / torchscript",
                difficulty="Medium",
                evidence_status=infer_status,
                evidence_source=comp_infer.evidence_source if comp_infer else None,
                supporting_papers=[primary_paper_name],
            )
        )
        p7_tasks.append(
            RoadmapTask(
                phase=7,
                task="Verify mathematical parity between eager PyTorch and compiled ONNX forward pass (max diff <= 1e-4)",
                description="Assert tensor output numerical parity between eager PyTorch and compiled runtime.",
                dependency=f"Export verified PyTorch model checkpoint to ONNX / TorchScript format ({infer_val})",
                input="Eager vs compiled outputs",
                output="Numerical equivalence audit report",
                technology="numpy.testing.assert_allclose",
                difficulty="Low",
                evidence_status="INFERRED",
                evidence_source="ResearchMind Performance Verification",
                supporting_papers=[primary_paper_name],
            )
        )

        if target_env == "edge" or is_limited_gpu:
            p7_tasks.append(
                RoadmapTask(
                    phase=7,
                    task="Apply INT8 Post-Training Quantization (PTQ) and operator fusion to achieve <= 50ms latency",
                    description="Quantize weights to INT8 to meet edge latency constraints.",
                    dependency=f"Export verified PyTorch model checkpoint to ONNX / TorchScript format ({infer_val})",
                    input="FP32 ONNX model",
                    output="Quantized INT8 model engine",
                    technology="onnxruntime.quantization / tflite",
                    difficulty="Medium",
                    evidence_status="INFERRED",
                    evidence_source=f"Constraint: target_environment={target_env}",
                    supporting_papers=[primary_paper_name],
                )
            )
            p7_tasks.append(
                RoadmapTask(
                    phase=7,
                    task="Profile memory consumption to guarantee peak VRAM remains under budget",
                    description="Profile peak allocated memory during continuous batch scoring.",
                    dependency="Apply INT8 Post-Training Quantization (PTQ) and operator fusion to achieve <= 50ms latency",
                    input="INT8 model engine",
                    output="VRAM memory footprint profile",
                    technology="memory_profiler",
                    difficulty="Low",
                    evidence_status="INFERRED",
                    evidence_source="ResearchMind Hardware Verification",
                    supporting_papers=[primary_paper_name],
                )
            )
        else:
            p7_tasks.append(
                RoadmapTask(
                    phase=7,
                    task=f"Build TensorRT / ONNX Runtime engine with dynamic shape batching for {infer_val}",
                    description="Compile high-throughput engine with dynamic tensor shapes.",
                    dependency=f"Export verified PyTorch model checkpoint to ONNX / TorchScript format ({infer_val})",
                    input="ONNX model",
                    output="TensorRT engine plan file",
                    technology="tensorrt / onnxruntime",
                    difficulty="Medium",
                    evidence_status="INFERRED",
                    evidence_source="High-Throughput Serving Spec",
                    supporting_papers=[primary_paper_name],
                )
            )

        p7_tasks.append(
            RoadmapTask(
                phase=7,
                task=f"Profile P50, P95, and P99 inference latencies across batch sizes (1, 4, 16, 64) for {infer_val}",
                description="Benchmark throughput and end-to-end response time across multiple concurrency levels.",
                dependency=f"Export verified PyTorch model checkpoint to ONNX / TorchScript format ({infer_val})",
                input="Compiled engine",
                output="Latency scorecard report",
                technology="triton perf_analyzer / timeit",
                difficulty="Low",
                evidence_status="INFERRED",
                evidence_source="ResearchMind Performance Verification",
                supporting_papers=[primary_paper_name],
            )
        )

        if comp_infer:
            p7_evidence.append({
                "component": "Inference",
                "value": infer_val,
                "source_paper": comp_infer.source_paper,
                "evidence": comp_infer.evidence,
                "status": comp_infer.status.value.lower(),
            })

        p7_phase = RoadmapPhase(
            phase=7,
            title="Phase 7: Inference Optimization, Quantization & Latency Profiling",
            description=f"Compile neural models for deployment ({infer_val}), apply quantization and kernel operator fusion, and benchmark inference latency.",
            tasks=[t.task for t in p7_tasks],
            task_items=p7_tasks,
            dependencies=["Phase 6: Empirical Evaluation, Benchmark Validation & Ablations"],
            inputs=["Verified model checkpoint from Phase 6", "Target latency / VRAM constraints"],
            outputs=["Compiled inference model (.onnx / TensorRT engine)", "Latency and throughput benchmark report", "Operator compatibility audit"],
            estimated_difficulty="Medium",
            evidence=p7_evidence,
            status="planned",
        )

        # =================================================================
        # PHASE 8 — Serving, API & Containerized Deployment
        # =================================================================
        p8_tasks: List[RoadmapTask] = []
        p8_evidence: List[Dict[str, Any]] = []

        backend_val = comp_backend.value if comp_backend else "FastAPI Async Microservice"
        backend_status = comp_backend.evidence_status.value if comp_backend else "INFERRED"
        frontend_val = comp_frontend.value if comp_frontend else "Interactive Web Dashboard"
        deploy_val = comp_deploy.value if comp_deploy else "Docker Container on Linux Host"
        deploy_status = comp_deploy.evidence_status.value if comp_deploy else "INFERRED"

        p8_tasks.append(
            RoadmapTask(
                phase=8,
                task=f"Wrap inference engine in high-performance asynchronous FastAPI REST endpoints ({backend_val})",
                description=f"Build production API microservice: {backend_val} exposing REST endpoints. Evidence: {comp_backend.evidence if comp_backend else 'API specification'}.",
                dependency="Phase 7: Inference Optimization, Quantization & Latency Profiling",
                input="Compiled inference engine from Phase 7",
                output=f"FastAPI microservice application ({backend_val})",
                technology="FastAPI / Pydantic / Uvicorn",
                difficulty="Low",
                evidence_status=backend_status,
                evidence_source=comp_backend.evidence_source if comp_backend else None,
                supporting_papers=[primary_paper_name],
            )
        )
        p8_tasks.append(
            RoadmapTask(
                phase=8,
                task="Implement input validation schemas via Pydantic and batching queue worker",
                description="Enforce strict payload typing and validation for incoming inference requests.",
                dependency=f"Wrap inference engine in high-performance asynchronous FastAPI REST endpoints ({backend_val})",
                input="API payload models",
                output="Pydantic request/response validators",
                technology="Pydantic v2",
                difficulty="Low",
                evidence_status="INFERRED",
                evidence_source="ResearchMind API Framework",
                supporting_papers=[primary_paper_name],
            )
        )
        p8_tasks.append(
            RoadmapTask(
                phase=8,
                task=f"Develop interactive client dashboard: {frontend_val}",
                description=f"Integrate interactive web UI ({frontend_val}) for real-time inference telemetry and result visualization.",
                dependency="Implement input validation schemas via Pydantic and batching queue worker",
                input="FastAPI REST endpoints and WebSocket channels",
                output="Interactive client frontend application",
                technology="React / Vite / Tailwind",
                difficulty="Low",
                evidence_status=comp_frontend.evidence_status.value if comp_frontend else "INFERRED",
                evidence_source=comp_frontend.evidence_source if comp_frontend else None,
                supporting_papers=[primary_paper_name],
            )
        )
        p8_tasks.append(
            RoadmapTask(
                phase=8,
                task=f"Write multi-stage production Dockerfile packaging runtime dependencies, model weights, and ASGI server ({deploy_val})",
                description=f"Package service into production container: {deploy_val}. Evidence: {comp_deploy.evidence if comp_deploy else 'Container specification'}.",
                dependency=f"Develop interactive client dashboard: {frontend_val}",
                input="FastAPI application and environment lockfiles",
                output=f"Multi-stage Docker container image ({deploy_val})",
                technology="Docker / Kubernetes",
                difficulty="Medium",
                evidence_status=deploy_status,
                evidence_source=comp_deploy.evidence_source if comp_deploy else None,
                supporting_papers=[primary_paper_name],
            )
        )
        p8_tasks.append(
            RoadmapTask(
                phase=8,
                task="Configure health check endpoints (/health), readiness probes, and Prometheus telemetry metrics",
                description="Instrument microservice with telemetry, Prometheus counters, and health check probes.",
                dependency=f"Write multi-stage production Dockerfile packaging runtime dependencies, model weights, and ASGI server ({deploy_val})",
                input="Container service",
                output="Prometheus telemetry instrumentation",
                technology="Prometheus / FastAPI middleware",
                difficulty="Low",
                evidence_status="INFERRED",
                evidence_source="ResearchMind Observability Framework",
                supporting_papers=[primary_paper_name],
            )
        )

        if open_source_only:
            p8_tasks.append(
                RoadmapTask(
                    phase=8,
                    task="Enforce constraint: Deploy strictly on open-source Linux container runtime with zero proprietary commercial dependencies",
                    description="Audit deployment manifests to ensure strictly open-source container engine.",
                    dependency="Write multi-stage production Dockerfile",
                    input="Deployment manifests",
                    output="Open-source deployment audit certificate",
                    technology="containerd / k8s",
                    difficulty="Low",
                    evidence_status="INFERRED",
                    evidence_source="Constraint: open_source_only",
                    supporting_papers=[primary_paper_name],
                )
            )

        if comp_backend:
            p8_evidence.append({
                "component": "Backend/API",
                "value": backend_val,
                "source_paper": comp_backend.source_paper,
                "evidence": comp_backend.evidence,
                "status": comp_backend.status.value.lower(),
            })

        p8_phase = RoadmapPhase(
            phase=8,
            title="Phase 8: API Microservice Integration, Containerization & Deployment",
            description=f"Build production FastAPI microservices ({backend_val}) around optimized inference runtimes, package Docker containers ({deploy_val}), and deploy monitoring.",
            tasks=[t.task for t in p8_tasks],
            task_items=p8_tasks,
            dependencies=["Phase 7: Inference Optimization, Quantization & Latency Profiling"],
            inputs=["Compiled inference artifact from Phase 7", "API request/response schema specifications"],
            outputs=["FastAPI production service application", "Multi-stage Docker container image", "Deployment manifests & health probes"],
            estimated_difficulty="Low" if open_source_only else "Medium",
            evidence=p8_evidence,
            status="planned",
        )

        phases = [p1_phase, p2_phase, p3_phase, p4_phase, p5_phase, p6_phase, p7_phase, p8_phase]

        # -----------------------------------------------------------------
        # Directed Dependency Graph
        # -----------------------------------------------------------------
        dependency_graph = [
            RoadmapDependencyEdge(from_phase=1, from_title=p1_phase.title, to_phase=2, to_title=p2_phase.title, reason="Environment and libraries depend on research scope and benchmark specifications."),
            RoadmapDependencyEdge(from_phase=1, from_title=p1_phase.title, to_phase=3, to_title=p3_phase.title, reason="Data preparation requires acquired benchmark dataset files."),
            RoadmapDependencyEdge(from_phase=2, from_title=p2_phase.title, to_phase=3, to_title=p3_phase.title, reason="Data preprocessing requires active virtual environment and data manipulation packages."),
            RoadmapDependencyEdge(from_phase=2, from_title=p2_phase.title, to_phase=4, to_title=p4_phase.title, reason="Model implementation requires deep learning framework and verified repositories."),
            RoadmapDependencyEdge(from_phase=3, from_title=p3_phase.title, to_phase=5, to_title=p5_phase.title, reason="Model training requires preprocessed training and validation data loaders."),
            RoadmapDependencyEdge(from_phase=4, from_title=p4_phase.title, to_phase=5, to_title=p5_phase.title, reason="Model training requires instantiated model architecture and layers."),
            RoadmapDependencyEdge(from_phase=5, from_title=p5_phase.title, to_phase=6, to_title=p6_phase.title, reason="Evaluation requires trained model weights and saved checkpoints."),
            RoadmapDependencyEdge(from_phase=6, from_title=p6_phase.title, to_phase=7, to_title=p7_phase.title, reason="Inference optimization requires verified model meeting target accuracy criteria."),
            RoadmapDependencyEdge(from_phase=7, from_title=p7_phase.title, to_phase=8, to_title=p8_phase.title, reason="Production microservice deployment requires compiled, quantized inference engine."),
        ]

        # Aggregate evidence across all phases
        all_evidence: List[Dict[str, Any]] = []
        for p in phases:
            all_evidence.extend(p.evidence)

        # Legacy milestones for DB backwards compatibility
        legacy_stages = [
            RoadmapMilestone(
                id="ms_dataset_preparation",
                stage=RoadmapStageType.DATASET_PREPARATION,
                title=p1_phase.title,
                description=p1_phase.description,
                prerequisites=[],
                estimated_effort="2-4 days",
                deliverables=p1_phase.outputs,
                supporting_resources=[
                    SupportingResource(
                        resource_type="paper",
                        title=d.title,
                        url=d.url,
                        identifier=d.doi,
                        document_id=d.internal_id,
                        relevance_note="Primary grounding publication",
                    )
                    for d in documents[:3]
                ],
            ),
            RoadmapMilestone(
                id="ms_baseline",
                stage=RoadmapStageType.BASELINE,
                title=p2_phase.title,
                description=p2_phase.description,
                prerequisites=["ms_dataset_preparation"],
                estimated_effort="1-2 days",
                deliverables=p2_phase.outputs,
                supporting_resources=[],
            ),
            RoadmapMilestone(
                id="ms_model_training",
                stage=RoadmapStageType.MODEL_TRAINING,
                title=p5_phase.title,
                description=p5_phase.description,
                prerequisites=["ms_baseline"],
                estimated_effort="3-5 days" if not is_limited_time else "1-2 days",
                deliverables=p5_phase.outputs,
                supporting_resources=[],
            ),
            RoadmapMilestone(
                id="ms_evaluation",
                stage=RoadmapStageType.EVALUATION,
                title=p6_phase.title,
                description=p6_phase.description,
                prerequisites=["ms_model_training"],
                estimated_effort="2-3 days",
                deliverables=p6_phase.outputs,
                supporting_resources=[],
            ),
            RoadmapMilestone(
                id="ms_backend",
                stage=RoadmapStageType.BACKEND,
                title=p7_phase.title,
                description=p7_phase.description,
                prerequisites=["ms_evaluation"],
                estimated_effort="2-3 days",
                deliverables=p7_phase.outputs,
                supporting_resources=[],
            ),
            RoadmapMilestone(
                id="ms_frontend_integration",
                stage=RoadmapStageType.FRONTEND_INTEGRATION,
                title="Frontend Integration & Interactive Telemetry",
                description="Connect web visualizers and telemetry streaming to the inference service.",
                prerequisites=["ms_backend"],
                estimated_effort="2-3 days",
                deliverables=["Interactive frontend dashboard", "Real-time telemetry display"],
                supporting_resources=[],
            ),
            RoadmapMilestone(
                id="ms_deployment",
                stage=RoadmapStageType.DEPLOYMENT,
                title=p8_phase.title,
                description=p8_phase.description,
                prerequisites=["ms_frontend_integration"],
                estimated_effort="1-2 days",
                deliverables=p8_phase.outputs,
                supporting_resources=[],
            ),
        ]

        if is_limited_time:
            total_duration = "3-5 days (Fast-Track Fine-Tuning)"
        elif is_limited_gpu:
            total_duration = "1-2 weeks (Constrained Memory Pipeline)"
        else:
            total_duration = "2-3 weeks (Full Engineering Implementation)"

        if is_scaffold:
            title = "Evidence insufficient — generic implementation scaffold"
            summary = "Evidence insufficient to synthesize paper-specific roadmap; generic scaffold presented."
        else:
            title = f"Implementation Roadmap: {primary_title}"
            summary = (
                f"Actionable 8-phase implementation roadmap for '{primary_title}'. "
                f"Directly derived from {len(documents)} research paper(s) and synthesized architecture '{model_val}'. "
                f"Grounds dataset ({ds_val}), preprocessing ({preproc_val}), model ({model_val}), "
                f"evaluation ({eval_val}), and serving ({infer_val})."
            )

        return ResearchRoadmapResponse(
            roadmap_id=roadmap_id,
            title=title,
            target_query=target_query,
            query=target_query,
            phases=phases,
            stages=legacy_stages,
            selected_paper_ids=selected_paper_ids,
            papers_used=selected_paper_ids,
            supporting_papers=selected_paper_ids,
            dependency_graph=dependency_graph,
            total_estimated_duration=total_duration,
            constraints_applied={
                "max_gpu_memory_gb": max_vram_gb,
                "limited_gpu": is_limited_gpu,
                "limited_time": is_limited_time,
                "public_datasets_only": public_datasets_only,
                "open_source_only": open_source_only,
                "target_environment": target_env,
            },
            evidence=all_evidence,
            architecture_hash=arch_hash,
            summary=summary,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

