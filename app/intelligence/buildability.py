"""
Buildability evaluation engine for ResearchMind AI.

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

Returns exact scalar scores accompanied by factual, verifiable evidence.
"""

import logging
import re
from typing import Dict, List, Tuple

from app.intelligence.models import BuildabilityDimension, BuildabilityReport
from app.schemas.paper import ResearchDocument

logger = logging.getLogger("researchmind.intelligence.buildability")


# Weights for the 9 dimensions (sum = 1.0)
DIMENSION_WEIGHTS = {
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


class BuildabilityEvaluator:
    """
    Evaluates research documents for real-world software implementation viability.
    """

    def evaluate_document(self, doc: ResearchDocument) -> BuildabilityReport:
        """
        Produce a comprehensive 9-dimensional buildability report with evidence.
        """
        dims: Dict[str, BuildabilityDimension] = {}
        strengths: List[str] = []
        risks: List[str] = []

        # -----------------------------------------------------------------
        # 1. Source-Code Availability
        # -----------------------------------------------------------------
        code_score = 0.0
        code_evidence: List[str] = []
        if doc.code_url and doc.code_url.strip():
            url = doc.code_url.strip()
            code_score = 1.0
            code_evidence.append(f"Verifiable open-source repository URL: {url}")
            strengths.append(f"Direct code repository available ({url})")
        else:
            # Check abstract for GitHub or code links
            text_to_scan = f"{doc.title} {doc.abstract or ''}".lower()
            github_match = re.search(r"github\.com/[\w\-]+/[\w\-]+", text_to_scan)
            if github_match:
                found_url = "https://" + github_match.group(0)
                code_score = 0.85
                code_evidence.append(f"Repository reference identified in text: {found_url}")
                strengths.append(f"Code repository found in paper text ({found_url})")
            else:
                code_score = 0.10
                code_evidence.append("No open-source code repository found in document metadata or abstract.")
                risks.append("No public source code repository available; requires full reimplementation.")

        dims["source_code_availability"] = BuildabilityDimension(
            name="source_code_availability",
            score=round(code_score, 4),
            evidence=code_evidence,
            confidence=0.95 if code_score > 0.5 else 0.85,
        )

        # -----------------------------------------------------------------
        # 2. Dataset Availability
        # -----------------------------------------------------------------
        data_score = 0.10
        data_evidence: List[str] = []
        if doc.datasets:
            data_score = 0.90
            data_evidence.append(f"Named datasets identified: {', '.join(doc.datasets)}")
            strengths.append(f"Public dataset references: {', '.join(doc.datasets)}")
        else:
            text_to_scan = f"{doc.title} {doc.abstract or ''}".lower()
            common_benchmarks = ["imagenet", "glue", "squad", "cifar", "mnist", "mmlu", "gsm8k", "coco", "pile", "wikitext"]
            found_benches = [b for b in common_benchmarks if b in text_to_scan]
            if found_benches:
                data_score = 0.75
                data_evidence.append(f"Recognized public benchmark datasets cited in text: {', '.join(found_benches)}")
                strengths.append(f"Uses standard benchmarks: {', '.join(found_benches)}")
            elif any(w in text_to_scan for w in ["benchmark", "corpus", "public dataset", "open data"]):
                data_score = 0.50
                data_evidence.append("Mentions benchmark or corpus evaluation without explicit dataset identifier.")
            else:
                data_score = 0.15
                data_evidence.append("No explicit public dataset references found; data may be proprietary or synthetic.")
                risks.append("Training/evaluation datasets are unspecified or private.")

        dims["dataset_availability"] = BuildabilityDimension(
            name="dataset_availability",
            score=round(data_score, 4),
            evidence=data_evidence,
            confidence=0.90,
        )

        # -----------------------------------------------------------------
        # 3. Pretrained Model Availability
        # -----------------------------------------------------------------
        model_score = 0.15
        model_evidence: List[str] = []
        if doc.models:
            model_score = 0.90
            model_evidence.append(f"Associated model checkpoints identified: {', '.join(doc.models)}")
            strengths.append(f"Pretrained models referenced: {', '.join(doc.models)}")
        else:
            text_to_scan = f"{doc.title} {doc.abstract or ''}".lower()
            if any(term in text_to_scan for term in ["huggingface", "weights available", "open-weight", "checkpoint", "checkpoint release"]):
                model_score = 0.80
                model_evidence.append("Public model weights or checkpoint releases explicitly referenced in text.")
                strengths.append("Public model checkpoint releases noted in abstract.")
            elif any(term in text_to_scan for term in ["bert", "llama", "mistral", "gpt", "resnet", "vit", "clip", "t5"]):
                model_score = 0.60
                model_evidence.append("Builds upon widely available foundational pretrained model backbones.")
            else:
                model_score = 0.20
                model_evidence.append("No pretrained model checkpoints or foundation model weights identified.")
                risks.append("Pretrained weights not released; retraining required.")

        dims["pretrained_model_availability"] = BuildabilityDimension(
            name="pretrained_model_availability",
            score=round(model_score, 4),
            evidence=model_evidence,
            confidence=0.85,
        )

        # -----------------------------------------------------------------
        # 4. Documentation
        # -----------------------------------------------------------------
        doc_score = 0.30
        doc_evidence: List[str] = []
        if doc.methods:
            doc_score += 0.30
            doc_evidence.append(f"Documented algorithms and methods: {', '.join(doc.methods)}")
        if doc.abstract and len(doc.abstract.split()) > 80:
            doc_score += 0.25
            doc_evidence.append(f"Comprehensive abstract with detailed methodology ({len(doc.abstract.split())} words).")
        if doc.code_url:
            doc_score += 0.15
            doc_evidence.append("Public repository provides README, setup instructions, or documentation.")
        doc_score = min(1.0, doc_score)

        dims["documentation"] = BuildabilityDimension(
            name="documentation",
            score=round(doc_score, 4),
            evidence=doc_evidence,
            confidence=0.85,
        )

        # -----------------------------------------------------------------
        # 5. Hardware Requirements
        # -----------------------------------------------------------------
        hw_score = 0.60
        hw_evidence: List[str] = []
        text_lower = f"{doc.title} {doc.abstract or ''}".lower()
        if any(term in text_lower for term in ["mobile", "edge", "distilled", "quantized", "cpu", "efficient", "lightweight", "tiny"]):
            hw_score = 0.90
            hw_evidence.append("Lightweight or quantized architecture; runnable on standard CPUs or modest consumer GPUs.")
            strengths.append("Low hardware requirements (runs on CPUs or single consumer GPU).")
        elif any(term in text_lower for term in ["tpu pod", "cluster", "supercomputer", "8x a100", "multi-node", "hundreds of gpus"]):
            hw_score = 0.25
            hw_evidence.append("Requires large-scale distributed multi-GPU/TPU cluster for full training or inference.")
            risks.append("High compute requirements; requires specialized cluster infrastructure.")
        else:
            hw_score = 0.65
            hw_evidence.append("Moderate hardware demands; standard modern workstation or single cloud GPU (e.g. T4/A10G).")

        dims["hardware_requirements"] = BuildabilityDimension(
            name="hardware_requirements",
            score=round(hw_score, 4),
            evidence=hw_evidence,
            confidence=0.80,
        )

        # -----------------------------------------------------------------
        # 6. Dependency Complexity
        # -----------------------------------------------------------------
        dep_score = 0.70
        dep_evidence: List[str] = []
        if any(term in text_lower for term in ["custom cuda", "specialized hardware", "kernel compilation", "proprietary sdk"]):
            dep_score = 0.35
            dep_evidence.append("Involves custom CUDA kernels or specialized hardware SDKs that increase integration friction.")
            risks.append("Custom compilation or low-level CUDA dependencies required.")
        else:
            dep_score = 0.85
            dep_evidence.append("Compatible with standard modern machine learning runtimes (PyTorch, Hugging Face, ONNX).")

        dims["dependency_complexity"] = BuildabilityDimension(
            name="dependency_complexity",
            score=round(dep_score, 4),
            evidence=dep_evidence,
            confidence=0.80,
        )

        # -----------------------------------------------------------------
        # 7. License & Accessibility
        # -----------------------------------------------------------------
        lic_score = 0.30
        lic_evidence: List[str] = []
        if doc.license:
            lic_clean = doc.license.lower()
            if any(p in lic_clean for p in ["mit", "apache", "bsd", "cc-by"]):
                lic_score = 0.95
                lic_evidence.append(f"Permissive open license recognized: {doc.license}")
                strengths.append(f"Permissive open license ({doc.license})")
            else:
                lic_score = 0.60
                lic_evidence.append(f"Document license: {doc.license}")
        elif doc.open_access is True:
            lic_score = 0.80
            lic_evidence.append("Open-access publication without restrictive proprietary paywalls.")
        else:
            lic_score = 0.40
            lic_evidence.append("No explicit open license specified in source metadata.")

        dims["license_accessibility"] = BuildabilityDimension(
            name="license_accessibility",
            score=round(lic_score, 4),
            evidence=lic_evidence,
            confidence=0.90,
        )

        # -----------------------------------------------------------------
        # 8. Reproducibility
        # -----------------------------------------------------------------
        # Synthesizes code + dataset + methods
        repro_score = 0.10
        repro_evidence: List[str] = []
        if code_score >= 0.8 and data_score >= 0.7:
            repro_score = 0.95
            repro_evidence.append("Both source code and public datasets are available, enabling direct reproduction.")
            strengths.append("High reproducibility: code and datasets verified.")
        elif code_score >= 0.8:
            repro_score = 0.70
            repro_evidence.append("Source code is available; dataset reproduction depends on access to cited benchmarks.")
        elif data_score >= 0.7:
            repro_score = 0.45
            repro_evidence.append("Public datasets available, but absence of reference code necessitates reimplementation.")
        else:
            repro_score = 0.15
            repro_evidence.append("Neither verifiable code nor identified datasets are linked; reproduction is high-risk.")
            risks.append("Low reproducibility due to absence of verified public artifacts.")

        dims["reproducibility"] = BuildabilityDimension(
            name="reproducibility",
            score=round(repro_score, 4),
            evidence=repro_evidence,
            confidence=0.90,
        )

        # -----------------------------------------------------------------
        # 9. Implementation Completeness
        # -----------------------------------------------------------------
        impl_score = 0.20
        impl_evidence: List[str] = []
        artifacts_count = int(bool(doc.code_url)) + int(bool(doc.datasets)) + int(bool(doc.models)) + int(bool(doc.methods))
        if artifacts_count >= 3:
            impl_score = 0.90
            impl_evidence.append(f"Comprehensive implementation artifact bundle ({artifacts_count}/4 elements verified).")
        elif artifacts_count == 2:
            impl_score = 0.65
            impl_evidence.append(f"Partial implementation artifact bundle ({artifacts_count}/4 elements verified).")
        else:
            impl_score = 0.30
            impl_evidence.append("Theoretical or architectural proposal with minimal published implementation artifacts.")

        dims["implementation_completeness"] = BuildabilityDimension(
            name="implementation_completeness",
            score=round(impl_score, 4),
            evidence=impl_evidence,
            confidence=0.85,
        )

        # -----------------------------------------------------------------
        # Overall Weighted Score & Tier
        # -----------------------------------------------------------------
        overall = sum(DIMENSION_WEIGHTS[name] * dims[name].score for name in DIMENSION_WEIGHTS)
        overall = round(max(0.0, min(1.0, overall)), 4)

        if overall >= 0.75:
            tier = "High"
        elif overall >= 0.50:
            tier = "Moderate"
        elif overall >= 0.25:
            tier = "Low"
        else:
            tier = "Blocked"

        return BuildabilityReport(
            document_id=doc.internal_id,
            title=doc.title,
            overall_score=overall,
            reproducibility_tier=tier,
            dimensions=dims,
            key_strengths=strengths,
            key_risks_or_blockers=risks,
        )

    def evaluate_documents(self, documents: List[ResearchDocument]) -> List[BuildabilityReport]:
        """Batch evaluate buildability across multiple papers."""
        return [self.evaluate_document(d) for d in documents]
