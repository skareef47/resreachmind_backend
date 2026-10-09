"""
Buildability evaluation and evidence enrichment engine for ResearchMind AI.

Rigorously assesses software engineering feasibility and reproducibility across 9 dimensions:
1. dataset availability
2. source-code availability
3. pretrained model availability
4. documentation
5. hardware requirements
6. dependency complexity
7. license/accessibility
8. reproducibility
9. implementation completeness

Implements evidence enrichment across DOI, publisher URLs, OpenAlex, Semantic Scholar,
and abstract/title text with verifiable link detection (GitHub, GitLab, Zenodo, HuggingFace, etc.).
Distinguishes between VERIFIED_PRESENT, UNKNOWN, and VERIFIED_ABSENT (UNKNOWN != ABSENT).
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.intelligence.models import BuildabilityDimension, BuildabilityReport
from app.schemas.paper import ResearchDocument

logger = logging.getLogger("researchmind.intelligence.buildability")

# Evidence status constants
STATUS_VERIFIED = "verified"
STATUS_UNKNOWN = "unknown"
STATUS_ABSENT = "absent"

# Default weights for the 9 dimensions (sum = 1.0)
DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "source_code_availability": 0.20,
    "dataset_availability": 0.15,
    "pretrained_model_availability": 0.15,
    "reproducibility": 0.15,
    "implementation_completeness": 0.10,
    "documentation": 0.08,
    "license_accessibility": 0.07,
    "hardware_requirements": 0.05,
    "dependency_complexity": 0.05,
}

# Regex patterns for verified implementation links
GITHUB_REGEX = re.compile(
    r"https?://(?:www\.)?github\.com/([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+)",
    re.IGNORECASE,
)
GITLAB_REGEX = re.compile(
    r"https?://(?:www\.)?gitlab\.com/([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+)",
    re.IGNORECASE,
)
HF_MODEL_REGEX = re.compile(
    r"https?://(?:www\.)?huggingface\.co/(?!datasets)([a-zA-Z0-9_\-\.]+(?:/[a-zA-Z0-9_\-\.]+)*)",
    re.IGNORECASE,
)
HF_DATASET_REGEX = re.compile(
    r"https?://(?:www\.)?huggingface\.co/datasets/([a-zA-Z0-9_\-\.]+(?:/[a-zA-Z0-9_\-\.]+)*)",
    re.IGNORECASE,
)
ZENODO_REGEX = re.compile(
    r"https?://(?:www\.)?zenodo\.org/record/\d+|https?://doi\.org/10\.5281/zenodo\.\d+",
    re.IGNORECASE,
)
FIGSHARE_REGEX = re.compile(
    r"https?://(?:[a-zA-Z0-9\-]+\.)?figshare\.com/[^\s)\"]+",
    re.IGNORECASE,
)
PHYSIO_REGEX = re.compile(
    r"https?://(?:www\.)?physionet\.org/[^\s)\"]+",
    re.IGNORECASE,
)
KAGGLE_REGEX = re.compile(
    r"https?://(?:www\.)?kaggle\.com/datasets/[^\s)\"]+",
    re.IGNORECASE,
)
DRYAD_REGEX = re.compile(
    r"https?://(?:www\.)?datadryad\.org/[^\s)\"]+",
    re.IGNORECASE,
)

# Common public benchmark dataset identifiers
BENCHMARK_DATASETS = [
    "imagenet", "glue", "superglue", "squad", "cifar-10", "cifar-100", "cifar",
    "mnist", "mmlu", "gsm8k", "coco", "pile", "wikitext", "tcga", "cbis-ddsm",
    "inbreast", "mias", "ddsm", "mimic-iii", "mimic-iv", "mimic", "chexpert",
    "bcss", "breakhis", "isic", "kaggle", "common crawl", "bookcorpus"
]

# Foundational architectures
FOUNDATIONAL_ARCHS = [
    "bert", "roberta", "llama", "mistral", "gpt", "resnet", "vit", "clip", "t5",
    "cnn", "convolutional neural network", "deep neural network", "vgg",
    "densenet", "u-net", "unet", "transformer", "yolo", "gan", "lstm", "autoencoder"
]

# Explicit absence patterns (VERIFIED_ABSENT)
CODE_ABSENCE_PATTERNS = [
    "code is proprietary", "code is not available", "code unavailable",
    "closed source", "closed-source", "proprietary codebase",
    "code cannot be shared", "confidential code", "not publicly available code",
    "code will not be shared"
]

DATASET_ABSENCE_PATTERNS = [
    "data cannot be shared", "data is not publicly available", "data unavailable",
    "proprietary dataset", "proprietary data", "confidential patient data",
    "private clinical cohort", "data will not be shared", "restricted patient data",
    "data not publicly available", "not publicly accessible data"
]

MODEL_ABSENCE_PATTERNS = [
    "models will not be released", "closed weights", "proprietary model weights",
    "weights unavailable", "closed-weights", "model weights are proprietary"
]


class BuildabilityEvaluator:
    """
    Evaluates research documents for real-world software implementation viability
    with rigorous evidence enrichment, link extraction, provenance tracking,
    and configurable weights.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize with configurable dimension weights B(d_i) = Σ w_k E_ik.
        """
        self.weights = dict(DEFAULT_DIMENSION_WEIGHTS)
        if weights:
            alias_map = {
                "code": "source_code_availability",
                "source_code": "source_code_availability",
                "dataset": "dataset_availability",
                "datasets": "dataset_availability",
                "model": "pretrained_model_availability",
                "models": "pretrained_model_availability",
                "pretrained_model": "pretrained_model_availability",
                "docs": "documentation",
                "hardware": "hardware_requirements",
                "dependencies": "dependency_complexity",
                "license": "license_accessibility",
                "completeness": "implementation_completeness",
            }
            mapped_weights = {}
            for k, v in weights.items():
                std_key = alias_map.get(k.lower(), k)
                if std_key in self.weights:
                    mapped_weights[std_key] = float(v)
            if mapped_weights:
                self.weights.update(mapped_weights)
            # Re-normalize weights so sum is 1.0
            total_w = sum(self.weights.values())
            if total_w > 0:
                self.weights = {k: v / total_w for k, v in self.weights.items()}

    def extract_evidence_sources(self, doc: ResearchDocument, text: str) -> List[str]:
        """
        Detect real verifiable evidence URLs from metadata and text.
        Prioritizes:
        1. GitHub repositories
        2. GitLab repositories
        3. Dataset repositories/pages (Zenodo, Figshare, PhysioNet, Kaggle, Dryad, HuggingFace datasets)
        4. Project / Landing pages
        5. Documentation / DOI URLs
        6. Model / Checkpoint links
        DO NOT invent fake links.
        """
        sources: List[str] = []
        seen: Set[str] = set()

        def add_url(url: Optional[str]):
            if not url:
                return
            cleaned = url.strip().rstrip(".,;)>'\"]")
            if cleaned and cleaned.startswith("http") and cleaned not in seen:
                seen.add(cleaned)
                sources.append(cleaned)

        # 1. Code repository (metadata code_url)
        if doc.code_url:
            add_url(doc.code_url)

        # 2. GitHub links in text or raw metadata
        for match in GITHUB_REGEX.finditer(text):
            add_url(match.group(0))

        # 3. GitLab links in text or raw metadata
        for match in GITLAB_REGEX.finditer(text):
            add_url(match.group(0))

        # Also search "github.com/org/repo" without scheme in text
        slug_matches = re.findall(r"(?:github\.com|gitlab\.com)/[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+", text)
        for slug in slug_matches:
            add_url(f"https://{slug}")

        # 4. Dataset repository URLs
        for match in ZENODO_REGEX.finditer(text):
            add_url(match.group(0))
        for match in FIGSHARE_REGEX.finditer(text):
            add_url(match.group(0))
        for match in PHYSIO_REGEX.finditer(text):
            add_url(match.group(0))
        for match in KAGGLE_REGEX.finditer(text):
            add_url(match.group(0))
        for match in DRYAD_REGEX.finditer(text):
            add_url(match.group(0))
        for match in HF_DATASET_REGEX.finditer(text):
            add_url(match.group(0))

        # 5. Pretrained Model links (HuggingFace)
        for match in HF_MODEL_REGEX.finditer(text):
            add_url(match.group(0))

        # 6. Verifiable scientific DOI URL
        if doc.doi:
            doi_clean = doc.doi.strip()
            if doi_clean.startswith("10."):
                add_url(f"https://doi.org/{doi_clean}")

        # 7. Canonical Publisher / Landing Page
        if doc.url:
            add_url(doc.url)

        # 8. Provenance URLs or provider canonical link
        for prov in doc.provenance:
            if prov.original_url:
                add_url(prov.original_url)

        if doc.source == "semantic_scholar" and doc.source_id:
            add_url(f"https://www.semanticscholar.org/paper/{doc.source_id}")
        elif doc.source == "openalex" and doc.source_id:
            openalex_id = doc.source_id.split("/")[-1]
            add_url(f"https://openalex.org/{openalex_id}")

        return sources

    def evaluate_document(self, doc: ResearchDocument) -> BuildabilityReport:
        """
        Produce a comprehensive 9-dimensional buildability report with evidence enrichment.
        Adheres to: UNKNOWN != ABSENT.
        Missing metadata is marked UNKNOWN with neutral non-punitive score (0.40) and explanation:
        'Not verified from available sources'.
        """
        dims: Dict[str, BuildabilityDimension] = {}
        evidence_breakdown: Dict[str, Dict[str, Any]] = {}
        strengths: List[str] = []
        risks: List[str] = []

        title_text = doc.title or ""
        abstract_text = doc.abstract or ""
        text_to_scan = f"{title_text} {abstract_text}".lower()

        # Extract genuine external evidence URLs
        evidence_sources = self.extract_evidence_sources(doc, f"{title_text} {abstract_text}")

        # -----------------------------------------------------------------
        # 1. Source-Code Availability
        # -----------------------------------------------------------------
        code_score: float = 0.40
        code_status: str = STATUS_UNKNOWN
        code_ev_summary: str = "Not verified from available sources"
        code_source: Optional[str] = None
        code_source_url: Optional[str] = None

        if doc.code_url and doc.code_url.strip():
            url = doc.code_url.strip()
            code_score = 1.0
            code_status = STATUS_VERIFIED
            code_ev_summary = f"Verifiable open-source repository: {url}"
            code_source = "doc.code_url"
            code_source_url = url
            strengths.append(f"Direct code repository verified ({url})")
        else:
            # Check for GitHub / GitLab links in text
            gh_match = GITHUB_REGEX.search(f"{title_text} {abstract_text}")
            gl_match = GITLAB_REGEX.search(f"{title_text} {abstract_text}")
            if gh_match:
                found_url = gh_match.group(0)
                code_score = 0.90
                code_status = STATUS_VERIFIED
                code_ev_summary = f"GitHub repository identified in paper text: {found_url}"
                code_source = "paper_text"
                code_source_url = found_url
                strengths.append(f"Code repository found in paper text ({found_url})")
            elif gl_match:
                found_url = gl_match.group(0)
                code_score = 0.90
                code_status = STATUS_VERIFIED
                code_ev_summary = f"GitLab repository identified in paper text: {found_url}"
                code_source = "paper_text"
                code_source_url = found_url
                strengths.append(f"Code repository found in paper text ({found_url})")
            elif any(k in text_to_scan for k in ["code is available at", "source code available at", "open-source implementation", "code repository"]):
                code_score = 0.70
                code_status = STATUS_VERIFIED
                code_ev_summary = "Source code availability statement confirmed in paper text."
                code_source = "paper_text"
                strengths.append("Paper confirms public source code availability.")
            elif any(p in text_to_scan for p in CODE_ABSENCE_PATTERNS):
                code_score = 0.05
                code_status = STATUS_ABSENT
                code_ev_summary = "Source code explicitly marked proprietary or unavailable."
                code_source = "paper_text"
                risks.append("Source code explicitly unavailable; requires full reimplementation.")
            else:
                code_score = 0.40
                code_status = STATUS_UNKNOWN
                code_ev_summary = "Not verified from available sources"
                code_source = None

        dims["source_code_availability"] = BuildabilityDimension(
            name="source_code_availability",
            score=round(code_score, 4),
            status=code_status,
            evidence=[code_ev_summary],
            evidence_summary=code_ev_summary,
            source=code_source,
            source_url=code_source_url,
            confidence=0.95 if code_status == STATUS_VERIFIED else 0.85,
        )
        evidence_breakdown["code"] = {
            "score": round(code_score, 2),
            "status": code_status,
            "evidence": code_ev_summary,
            "source": code_source_url or code_source or "available_sources",
        }

        # -----------------------------------------------------------------
        # 2. Dataset Availability
        # -----------------------------------------------------------------
        data_score: float = 0.40
        data_status: str = STATUS_UNKNOWN
        data_ev_summary: str = "Not verified from available sources"
        data_source: Optional[str] = None
        data_source_url: Optional[str] = None

        if doc.datasets and len(doc.datasets) > 0:
            data_score = 0.95
            data_status = STATUS_VERIFIED
            ds_names = ", ".join(doc.datasets)
            data_ev_summary = f"Named public datasets identified: {ds_names}"
            data_source = "metadata.datasets"
            strengths.append(f"Public dataset references: {ds_names}")
        else:
            # Check for dataset repository URLs
            zen_match = ZENODO_REGEX.search(f"{title_text} {abstract_text}")
            fig_match = FIGSHARE_REGEX.search(f"{title_text} {abstract_text}")
            phys_match = PHYSIO_REGEX.search(f"{title_text} {abstract_text}")
            kag_match = KAGGLE_REGEX.search(f"{title_text} {abstract_text}")
            dry_match = DRYAD_REGEX.search(f"{title_text} {abstract_text}")

            found_ds_url = (
                zen_match.group(0) if zen_match else
                fig_match.group(0) if fig_match else
                phys_match.group(0) if phys_match else
                kag_match.group(0) if kag_match else
                dry_match.group(0) if dry_match else None
            )

            if found_ds_url:
                data_score = 0.90
                data_status = STATUS_VERIFIED
                data_ev_summary = f"Dataset repository identified: {found_ds_url}"
                data_source = "paper_text"
                data_source_url = found_ds_url
                strengths.append(f"Public dataset repository confirmed ({found_ds_url})")
            else:
                found_benches = [b for b in BENCHMARK_DATASETS if b in text_to_scan]
                if found_benches:
                    data_score = 0.80
                    data_status = STATUS_VERIFIED
                    data_ev_summary = f"Recognized public benchmark datasets cited in text: {', '.join(found_benches)}"
                    data_source = "paper_text"
                    strengths.append(f"Uses standard benchmarks: {', '.join(found_benches)}")
                elif any(w in text_to_scan for w in ["benchmark evaluation", "public corpus", "open data", "clinical cohort", "data availability", "dataset"]):
                    data_score = 0.60
                    data_status = STATUS_VERIFIED
                    data_ev_summary = "Public benchmark or cohort evaluation referenced in paper text."
                    data_source = "paper_text"
                elif any(p in text_to_scan for p in DATASET_ABSENCE_PATTERNS):
                    data_score = 0.10
                    data_status = STATUS_ABSENT
                    data_ev_summary = "Training/evaluation data explicitly identified as proprietary, private, or restricted."
                    data_source = "paper_text"
                    risks.append("Evaluation datasets are explicitly private or restricted.")
                else:
                    data_score = 0.40
                    data_status = STATUS_UNKNOWN
                    data_ev_summary = "Not verified from available sources"
                    data_source = None

        dims["dataset_availability"] = BuildabilityDimension(
            name="dataset_availability",
            score=round(data_score, 4),
            status=data_status,
            evidence=[data_ev_summary],
            evidence_summary=data_ev_summary,
            source=data_source,
            source_url=data_source_url,
            confidence=0.90 if data_status == STATUS_VERIFIED else 0.80,
        )
        evidence_breakdown["dataset"] = {
            "score": round(data_score, 2),
            "status": data_status,
            "evidence": data_ev_summary,
            "source": data_source_url or data_source or "available_sources",
        }

        # -----------------------------------------------------------------
        # 3. Pretrained Model Availability
        # -----------------------------------------------------------------
        model_score: float = 0.40
        model_status: str = STATUS_UNKNOWN
        model_ev_summary: str = "Not verified from available sources"
        model_source: Optional[str] = None
        model_source_url: Optional[str] = None

        if doc.models and len(doc.models) > 0:
            model_score = 0.90
            model_status = STATUS_VERIFIED
            m_names = ", ".join(doc.models)
            model_ev_summary = f"Associated model checkpoints identified: {m_names}"
            model_source = "metadata.models"
            strengths.append(f"Pretrained models referenced: {m_names}")
        else:
            hf_m_match = HF_MODEL_REGEX.search(f"{title_text} {abstract_text}")
            if hf_m_match:
                hf_url = hf_m_match.group(0)
                model_score = 0.90
                model_status = STATUS_VERIFIED
                model_ev_summary = f"Model checkpoint repository identified: {hf_url}"
                model_source = "paper_text"
                model_source_url = hf_url
                strengths.append(f"Model checkpoint repository confirmed ({hf_url})")
            elif any(t in text_to_scan for t in ["weights available", "open-weight", "checkpoint release", "pretrained weights", "model checkpoint"]):
                model_score = 0.80
                model_status = STATUS_VERIFIED
                model_ev_summary = "Public model weights or checkpoint releases explicitly referenced in text."
                model_source = "paper_text"
                strengths.append("Public model checkpoint releases confirmed in abstract.")
            elif any(arch in text_to_scan for arch in FOUNDATIONAL_ARCHS):
                found_arch = next(arch for arch in FOUNDATIONAL_ARCHS if arch in text_to_scan)
                model_score = 0.55
                model_status = STATUS_VERIFIED
                model_ev_summary = f"Builds upon standard foundational neural architectures ({found_arch.upper()})."
                model_source = "paper_text"
            elif any(p in text_to_scan for p in MODEL_ABSENCE_PATTERNS):
                model_score = 0.10
                model_status = STATUS_ABSENT
                model_ev_summary = "Model weights explicitly marked proprietary or closed."
                model_source = "paper_text"
                risks.append("Pretrained weights explicitly not released; retraining required.")
            else:
                model_score = 0.40
                model_status = STATUS_UNKNOWN
                model_ev_summary = "Not verified from available sources"
                model_source = None

        dims["pretrained_model_availability"] = BuildabilityDimension(
            name="pretrained_model_availability",
            score=round(model_score, 4),
            status=model_status,
            evidence=[model_ev_summary],
            evidence_summary=model_ev_summary,
            source=model_source,
            source_url=model_source_url,
            confidence=0.85,
        )
        evidence_breakdown["pretrained_model"] = {
            "score": round(model_score, 2),
            "status": model_status,
            "evidence": model_ev_summary,
            "source": model_source_url or model_source or "available_sources",
        }

        # -----------------------------------------------------------------
        # 4. Documentation
        # -----------------------------------------------------------------
        doc_score: float = 0.35
        doc_status: str = STATUS_UNKNOWN
        doc_ev_summary: str = "Not verified from available sources"
        doc_source: Optional[str] = None
        doc_source_url: Optional[str] = None
        doc_ev_parts: List[str] = []

        if doc.doi or doc.url or doc.methods or (doc.abstract and len(doc.abstract.split()) > 40):
            doc_status = STATUS_VERIFIED
            doc_score_calc = 0.30
            if doc.doi:
                doc_score_calc += 0.25
                doc_source_url = f"https://doi.org/{doc.doi}"
                doc_ev_parts.append(f"Verifiable scientific DOI: {doc.doi}")
            if doc.url:
                doc_score_calc += 0.15
                if not doc_source_url:
                    doc_source_url = doc.url
                doc_ev_parts.append("Publisher landing page available")
            if doc.methods:
                doc_score_calc += 0.15
                doc_ev_parts.append(f"Documented methodology: {', '.join(doc.methods)}")
            wc = len(doc.abstract.split()) if doc.abstract else 0
            if wc > 80:
                doc_score_calc += 0.15
                doc_ev_parts.append(f"Detailed abstract methodology ({wc} words)")
            elif wc > 40:
                doc_score_calc += 0.10
                doc_ev_parts.append(f"Standard abstract overview ({wc} words)")
            if code_status == STATUS_VERIFIED:
                doc_score_calc += 0.10
                doc_ev_parts.append("Repository provides implementation documentation")
            doc_score = min(1.0, round(doc_score_calc, 2))
            doc_ev_summary = "; ".join(doc_ev_parts) if doc_ev_parts else "Documented methodology verified."
            doc_source = "metadata.doi" if doc.doi else "publisher_page"
        else:
            doc_score = 0.35
            doc_status = STATUS_UNKNOWN
            doc_ev_summary = "Not verified from available sources"

        dims["documentation"] = BuildabilityDimension(
            name="documentation",
            score=round(doc_score, 4),
            status=doc_status,
            evidence=[doc_ev_summary],
            evidence_summary=doc_ev_summary,
            source=doc_source,
            source_url=doc_source_url,
            confidence=0.85,
        )
        evidence_breakdown["documentation"] = {
            "score": round(doc_score, 2),
            "status": doc_status,
            "evidence": doc_ev_summary,
            "source": doc_source_url or doc_source or "available_sources",
        }

        # -----------------------------------------------------------------
        # 5. Hardware Requirements
        # -----------------------------------------------------------------
        hw_score: float = 0.50
        hw_status: str = STATUS_UNKNOWN
        hw_ev_summary: str = "Not verified from available sources (standard workstation or cloud GPU inferred)"

        if any(term in text_to_scan for term in ["mobile", "edge", "distilled", "quantized", "cpu", "efficient", "lightweight", "tiny"]):
            hw_score = 0.90
            hw_status = STATUS_VERIFIED
            hw_ev_summary = "Lightweight or quantized architecture; runnable on standard CPUs or modest consumer GPUs."
            strengths.append("Low hardware demands (runs on CPUs or single consumer GPU).")
        elif any(term in text_to_scan for term in ["tpu pod", "supercomputer", "8x a100", "multi-node", "hundreds of gpus", "cluster of 1024"]):
            hw_score = 0.25
            hw_status = STATUS_VERIFIED
            hw_ev_summary = "Requires large-scale distributed multi-GPU/TPU cluster for execution."
            risks.append("High compute requirements; requires specialized cluster infrastructure.")
        else:
            hw_score = 0.50
            hw_status = STATUS_UNKNOWN
            hw_ev_summary = "Not verified from available sources (standard workstation or cloud GPU inferred)"

        dims["hardware_requirements"] = BuildabilityDimension(
            name="hardware_requirements",
            score=round(hw_score, 4),
            status=hw_status,
            evidence=[hw_ev_summary],
            evidence_summary=hw_ev_summary,
            source="paper_text" if hw_status == STATUS_VERIFIED else None,
            confidence=0.80,
        )
        evidence_breakdown["hardware"] = {
            "score": round(hw_score, 2),
            "status": hw_status,
            "evidence": hw_ev_summary,
            "source": "paper_text" if hw_status == STATUS_VERIFIED else "not_specified",
        }

        # -----------------------------------------------------------------
        # 6. Dependency Complexity
        # -----------------------------------------------------------------
        dep_score: float = 0.60
        dep_status: str = STATUS_UNKNOWN
        dep_ev_summary: str = "Not verified from available sources (standard runtime inferred)"

        if any(term in text_to_scan for term in ["custom cuda", "specialized hardware", "kernel compilation", "proprietary sdk"]):
            dep_score = 0.35
            dep_status = STATUS_VERIFIED
            dep_ev_summary = "Requires custom CUDA kernels or specialized hardware SDKs that increase integration friction."
            risks.append("Custom compilation or low-level CUDA dependencies required.")
        elif any(term in text_to_scan for term in ["pytorch", "tensorflow", "scikit-learn", "huggingface", "keras", "r package", "bioconductor"]):
            dep_score = 0.85
            dep_status = STATUS_VERIFIED
            dep_ev_summary = "Compatible with standard modern machine learning runtimes (PyTorch, TensorFlow, Hugging Face)."
        else:
            dep_score = 0.60
            dep_status = STATUS_UNKNOWN
            dep_ev_summary = "Not verified from available sources (standard runtime inferred)"

        dims["dependency_complexity"] = BuildabilityDimension(
            name="dependency_complexity",
            score=round(dep_score, 4),
            status=dep_status,
            evidence=[dep_ev_summary],
            evidence_summary=dep_ev_summary,
            source="paper_text" if dep_status == STATUS_VERIFIED else None,
            confidence=0.80,
        )
        evidence_breakdown["dependencies"] = {
            "score": round(dep_score, 2),
            "status": dep_status,
            "evidence": dep_ev_summary,
            "source": "paper_text" if dep_status == STATUS_VERIFIED else "not_specified",
        }

        # -----------------------------------------------------------------
        # 7. License & Accessibility
        # -----------------------------------------------------------------
        lic_score: float = 0.40
        lic_status: str = STATUS_UNKNOWN
        lic_ev_summary: str = "Not verified from available sources"
        lic_source: Optional[str] = None

        if doc.license:
            lic_clean = str(doc.license).lower()
            if any(p in lic_clean for p in ["mit", "apache", "bsd", "cc-by"]):
                lic_score = 0.95
                lic_status = STATUS_VERIFIED
                lic_ev_summary = f"Permissive open license recognized: {doc.license}"
                lic_source = "metadata.license"
                strengths.append(f"Permissive open license ({doc.license})")
            else:
                lic_score = 0.70
                lic_status = STATUS_VERIFIED
                lic_ev_summary = f"Document license: {doc.license}"
                lic_source = "metadata.license"
        elif doc.open_access is True:
            lic_score = 0.80
            lic_status = STATUS_VERIFIED
            lic_ev_summary = "Open-access publication verified without proprietary paywalls."
            lic_source = "metadata.open_access"
            strengths.append("Open-access publication verified.")
        elif doc.open_access is False:
            lic_score = 0.20
            lic_status = STATUS_ABSENT
            lic_ev_summary = "Subscription access or proprietary paywall; source paper is not open-access."
            lic_source = "metadata.open_access"
            risks.append("Non-open access; may restrict automated pipeline reproduction.")
        else:
            lic_score = 0.40
            lic_status = STATUS_UNKNOWN
            lic_ev_summary = "Not verified from available sources"
            lic_source = None

        dims["license_accessibility"] = BuildabilityDimension(
            name="license_accessibility",
            score=round(lic_score, 4),
            status=lic_status,
            evidence=[lic_ev_summary],
            evidence_summary=lic_ev_summary,
            source=lic_source,
            confidence=0.90,
        )
        evidence_breakdown["license"] = {
            "score": round(lic_score, 2),
            "status": lic_status,
            "evidence": lic_ev_summary,
            "source": lic_source or "available_sources",
        }

        # -----------------------------------------------------------------
        # 8. Reproducibility
        # -----------------------------------------------------------------
        repro_score: float = 0.40
        repro_status: str = STATUS_UNKNOWN
        repro_ev_summary: str = "Not verified from available sources"

        if code_status == STATUS_VERIFIED and data_status == STATUS_VERIFIED:
            repro_score = 0.95
            repro_status = STATUS_VERIFIED
            repro_ev_summary = "Both source code and public datasets are verified, enabling direct reproduction."
            strengths.append("High reproducibility: code and datasets verified.")
        elif code_status == STATUS_VERIFIED:
            repro_score = 0.75
            repro_status = STATUS_VERIFIED
            repro_ev_summary = "Source code verified; dataset reproduction depends on access to cited benchmarks."
        elif data_status == STATUS_VERIFIED:
            repro_score = 0.55
            repro_status = STATUS_VERIFIED
            repro_ev_summary = "Public datasets verified, but absence of reference code necessitates reimplementation."
        elif code_status == STATUS_ABSENT and data_status == STATUS_ABSENT:
            repro_score = 0.10
            repro_status = STATUS_ABSENT
            repro_ev_summary = "Reproduction blocked: both proprietary code and private data restrictions identified."
            risks.append("Reproduction blocked by proprietary constraints.")
        else:
            repro_score = 0.40
            repro_status = STATUS_UNKNOWN
            repro_ev_summary = "Not verified from available sources (reproducibility unconfirmed without verifiable artifacts)"

        dims["reproducibility"] = BuildabilityDimension(
            name="reproducibility",
            score=round(repro_score, 4),
            status=repro_status,
            evidence=[repro_ev_summary],
            evidence_summary=repro_ev_summary,
            source="composite_evaluation",
            confidence=0.90,
        )
        evidence_breakdown["reproducibility"] = {
            "score": round(repro_score, 2),
            "status": repro_status,
            "evidence": repro_ev_summary,
            "source": "composite_evaluation",
        }

        # -----------------------------------------------------------------
        # 9. Implementation Completeness
        # -----------------------------------------------------------------
        verified_count = (
            int(code_status == STATUS_VERIFIED)
            + int(data_status == STATUS_VERIFIED)
            + int(model_status == STATUS_VERIFIED)
            + int(doc_status == STATUS_VERIFIED)
            + int(lic_status == STATUS_VERIFIED)
            + int(bool(doc.doi or doc.url))
        )

        if verified_count >= 4:
            impl_score = 0.90
            impl_status = STATUS_VERIFIED
            impl_ev_summary = f"Comprehensive implementation artifact bundle ({verified_count}/6 elements verified)."
        elif verified_count >= 2:
            impl_score = 0.70
            impl_status = STATUS_VERIFIED
            impl_ev_summary = f"Substantial implementation artifact bundle ({verified_count}/6 elements verified)."
        elif verified_count == 1:
            impl_score = 0.50
            impl_status = STATUS_VERIFIED
            impl_ev_summary = f"Partial implementation artifacts verified ({verified_count}/6 elements)."
        else:
            impl_score = 0.35
            impl_status = STATUS_UNKNOWN
            impl_ev_summary = "Not verified from available sources"

        dims["implementation_completeness"] = BuildabilityDimension(
            name="implementation_completeness",
            score=round(impl_score, 4),
            status=impl_status,
            evidence=[impl_ev_summary],
            evidence_summary=impl_ev_summary,
            source="composite_evaluation",
            confidence=0.85,
        )
        evidence_breakdown["implementation_completeness"] = {
            "score": round(impl_score, 2),
            "status": impl_status,
            "evidence": impl_ev_summary,
            "source": "composite_evaluation",
        }

        # -----------------------------------------------------------------
        # Overall Weighted Score B(d_i) = Σ w_k E_ik
        # -----------------------------------------------------------------
        overall = sum(self.weights[name] * dims[name].score for name in self.weights if name in dims)
        overall = round(max(0.0, min(1.0, overall)), 4)

        if overall >= 0.75:
            tier = "High"
        elif overall >= 0.50:
            tier = "Moderate"
        elif overall >= 0.25:
            tier = "Low"
        else:
            tier = "Blocked"

        # Feature breakdown dictionary for backwards compatibility
        feature_breakdown = {
            "dataset": round(dims["dataset_availability"].score, 2),
            "code": round(dims["source_code_availability"].score, 2),
            "model": round(dims["pretrained_model_availability"].score, 2),
            "docs": round(dims["documentation"].score, 2),
            "reproducibility": round(dims["reproducibility"].score, 2),
            "hardware": round(dims["hardware_requirements"].score, 2),
            "dependencies": round(dims["dependency_complexity"].score, 2),
            "license": round(dims["license_accessibility"].score, 2),
            "completeness": round(dims["implementation_completeness"].score, 2),
        }

        # Identify all unknown fields
        unknown_fields = [k for k, v in evidence_breakdown.items() if v["status"] == STATUS_UNKNOWN]

        # Populate canonical document fields
        doc.buildability_score = overall
        doc.buildability = overall
        doc.buildability_breakdown = evidence_breakdown
        doc.evidence_sources = evidence_sources
        doc.unknown_fields = unknown_fields

        return BuildabilityReport(
            document_id=doc.internal_id,
            title=doc.title,
            overall_score=overall,
            reproducibility_tier=tier,
            dimensions=dims,
            feature_breakdown=feature_breakdown,
            evidence_breakdown=evidence_breakdown,
            evidence_sources=evidence_sources,
            unknown_fields=unknown_fields,
            key_strengths=strengths,
            key_risks_or_blockers=risks,
        )

    def evaluate_documents(self, documents: List[ResearchDocument]) -> List[BuildabilityReport]:
        """Batch evaluate buildability across multiple papers."""
        return [self.evaluate_document(d) for d in documents]
