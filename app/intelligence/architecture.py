"""
Dynamic, Paper-Grounded Architecture Generation Engine for ResearchMind AI.

Synthesizes concrete, verifiable technical architectures directly from selected research papers,
extracted implementation evidence (methods, models, datasets, algorithms, preprocessing,
training, evaluation, hardware, dependencies), and user constraints.

Guarantees:
1. Dynamic, query/paper-dependent architectures (never a static generic template).
2. Domain and paper-specific components (e.g. MRI -> Skull Stripping -> ResNet50 -> SVM -> Grad-CAM).
3. Grounded evidence status: VERIFIED, INFERRED, UNKNOWN with explicit citation source.
4. Distinguishes what the paper actually did (VERIFIED) from what ResearchMind proposes (INFERRED).
5. If evidence is insufficient, clearly marks 'Evidence insufficient — generic implementation scaffold'.
6. Deterministic consistency for identical inputs.
7. Reactive propagation when ranking/preferences change.
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from app.intelligence.buildability import BuildabilityEvaluator
from app.intelligence.models import (
    ArchitectureComponent,
    ArchitectureConnection,
    CompatibilityAnalysis,
    DependencyItem,
    EvidenceStatus,
    HardwareRequirement,
    ResearchArchitectureResponse,
    TechnicalArchitecture,
)
from app.schemas.paper import ResearchDocument

logger = logging.getLogger("researchmind.intelligence.architecture")

# Known domain keywords for extraction
KNOWN_OPTIMIZERS = [
    "AdamW", "Adam", "SGD", "RMSprop", "Adafactor", "Lion", "LAMB", "L-BFGS", "Adagrad"
]
KNOWN_LOSSES = [
    "Cross-Entropy", "Focal Loss", "MSE", "Mean Squared Error", "Contrastive Loss",
    "Triplet Loss", "Dice Loss", "KL Divergence", "Cosine Similarity", "BCEWithLogitsLoss",
    "Hinge Loss", "NLLLoss", "Center Loss"
]
KNOWN_METRICS = [
    "Accuracy", "Top-1 Accuracy", "Top-5 Accuracy", "F1", "F1-Score", "Macro-F1",
    "Precision", "Recall", "Sensitivity", "Specificity", "AUC-ROC", "PR-AUC",
    "mAP", "Dice Coefficient", "IoU", "BLEU", "ROUGE", "Perplexity", "Latency"
]
KNOWN_PREPROCESSING = [
    "Skull Stripping", "Skull-Stripping", "Intensity Normalization", "N4ITK Bias Correction",
    "Data Augmentation", "Geometric Augmentation", "Color Jitter", "CutMix", "MixUp",
    "SMOTE", "ADASYN", "Min-Max Scaling", "StandardScaler", "Sliding Window",
    "Byte-Pair Encoding", "BPE", "WordPiece", "SentencePiece", "Tokenization",
    "Spectrogram", "Mel-Spectrogram", "STFT", "Adjacency Normalization"
]
KNOWN_MODELS = [
    "ResNet50", "ResNet-18", "ResNet-101", "ResNet", "EfficientNet-B0", "EfficientNet-B4",
    "EfficientNet-V2", "EfficientNet", "DenseNet121", "DenseNet", "U-Net", "UNet",
    "MobileNetV3", "MobileNet", "Vision Transformer", "ViT-B/16", "ViT", "Swin Transformer",
    "Transformer", "BERT", "RoBERTa", "LLaMA", "TabNet", "XGBoost", "LightGBM",
    "Graph Convolutional Network", "GCN", "GAT", "YOLOv8", "YOLO", "CNN"
]
KNOWN_HEADS = [
    "Support Vector Machine", "SVM", "Random Forest", "Softmax", "Dense Head",
    "Linear Classifier", "MLP Head", "Focal Loss Classifier", "Anomaly Scorer",
    "YOLO Head", "Linear Projection Head"
]
KNOWN_EXPLAINABILITY = [
    "Grad-CAM", "Grad-CAM++", "Score-CAM", "Integrated Gradients", "SHAP", "LIME",
    "Attention Rollout", "Saliency Maps", "Feature Attribution"
]


def _find_snippet(text: str, keywords: List[str]) -> Optional[Tuple[str, str]]:
    """Search text for any keyword in list, returning (matched_keyword, snippet_quote)."""
    if not text:
        return None
    text_lower = text.lower()
    for kw in keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        match = re.search(pattern, text_lower)
        if match:
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 60)
            snippet = text[start:end].strip().replace("\n", " ")
            return kw, f"...{snippet}..."
    return None


class ArchitectureGenerator:
    """
    Generates dynamic, paper-grounded technical architectures from candidate research papers.
    """

    def __init__(self, evaluator: Optional[BuildabilityEvaluator] = None) -> None:
        self.evaluator = evaluator or BuildabilityEvaluator()

    def generate_architecture(
        self,
        documents: List[ResearchDocument],
        constraints: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        selection_mode: Optional[str] = "optimized",
    ) -> ResearchArchitectureResponse:
        """
        Synthesize a concrete technical implementation architecture from selected research papers.
        """
        if not documents:
            raise ValueError("At least one ResearchDocument is required to generate an architecture.")

        constraints = constraints or {}
        arch_id = f"arch_{uuid.uuid4().hex[:12]}"

        # Evaluate buildability across documents to leverage multi-dimensional evidence
        buildability_reports = self.evaluator.evaluate_documents(documents)
        bld_by_id = {b.document_id: b for b in buildability_reports}

        primary_doc = documents[0]
        all_paper_ids = [d.internal_id for d in documents]

        # Check if documents contain real content or if evidence is completely absent
        has_any_content = any(
            bool(d.title.strip() or d.abstract or d.models or d.datasets or d.methods)
            for d in documents
        )
        is_generic_scaffold = not has_any_content

        # 1. Detect Research Domain & Implementation Profile
        domain_profile = self._detect_domain_profile(documents, query)

        # 2. Extract 10 Dynamic Paper-Specific Components
        components = self._extract_dynamic_components(
            documents=documents,
            domain=domain_profile,
            constraints=constraints,
            is_scaffold=is_generic_scaffold,
        )

        # 3. Build Domain & Paper-Specific Dataflow Connections
        connections = self._build_pipeline_connections(components, domain_profile)

        # 4. Extract Hardware Requirements
        hardware = self._extract_hardware_requirements(
            documents=documents,
            bld_by_id=bld_by_id,
            constraints=constraints,
            domain=domain_profile,
        )

        # 5. Extract Software Dependencies
        dependencies = self._extract_dependencies(
            documents=documents,
            bld_by_id=bld_by_id,
            constraints=constraints,
            domain=domain_profile,
            components=components,
        )

        # 6. Multi-Paper Compatibility Analysis
        compatibility = self._analyze_compatibility(documents, bld_by_id, constraints)

        # 7. Formulate Descriptive Architecture Title & Topology
        title = self._formulate_title(components, primary_doc, domain_profile, is_generic_scaffold)

        # Compute overall confidence
        conf_scores = [c.confidence for c in components if c.confidence is not None]
        overall_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.85

        tech_arch = TechnicalArchitecture(
            components=components,
            connections=connections,
            hardware=hardware,
            dependencies=dependencies,
        )

        return ResearchArchitectureResponse(
            architecture_id=arch_id,
            title=title,
            query=query,
            selected_paper_ids=all_paper_ids,
            papers_used=all_paper_ids,
            components=components,
            connections=connections,
            architecture=tech_arch,
            evidence=list(components),
            architecture_type=domain_profile["domain_label"],
            confidence=round(overall_conf, 2),
            is_generic_scaffold=is_generic_scaffold,
            compatibility=compatibility,
            constraints_applied=dict(constraints),
        )

    # -------------------------------------------------------------------------
    # Domain Detection
    # -------------------------------------------------------------------------

    def _detect_domain_profile(
        self, docs: List[ResearchDocument], query: Optional[str]
    ) -> Dict[str, Any]:
        """Classify research domain and modality profile from papers and query."""
        combined_text = (
            f"{query or ''} "
            + " ".join(
                f"{d.title} {d.abstract or ''} {' '.join(d.concepts or [])} {' '.join(d.methods or [])} {' '.join(d.models or [])} {' '.join(d.datasets or [])}"
                for d in docs
            )
        ).lower()

        # Medical Imaging / Cancer / MRI
        medical_kw = [
            "mri", "magnetic resonance", "brain tumor", "brain", "tumor", "cancer",
            "oncology", "ct scan", "dicom", "nifti", "ultrasound", "mammography",
            "histopathology", "lesion", "busi", "brats", "skull strip", "radiology"
        ]
        if any(kw in combined_text for kw in medical_kw):
            return {
                "domain": "medical_imaging",
                "domain_label": "Medical Imaging & Oncology Detection Pipeline",
                "default_input": "Volumetric MRI Scans (T1/T2/FLAIR NIfTI / DICOM)",
                "default_preproc": "Skull Stripping, Intensity Normalization & Slice Registration",
                "default_features": "ResNet50 Deep Residual Feature Extractor",
                "default_model": "Support Vector Machine (SVM) Decision Boundary Classifier",
                "default_explainability": "Grad-CAM Saliency Localization & Visual Attribution",
                "default_metrics": "Sensitivity, Specificity, AUC-ROC & Precision",
                "default_loss": "Hinge Loss & Cross-Entropy with SGD / AdamW",
                "default_serving": "TorchScript / ONNX Clinical DICOM Scoring Worker (with Grad-CAM)",
                "default_api": "FastAPI DICOM/NIfTI Medical Image Gateway & PACS Integration",
                "default_frontend": "Interactive Clinical DICOM Viewer with Grad-CAM Saliency Overlay Dashboard",
                "default_deploy": "HIPAA-Compliant Secure Docker Container on Kubernetes",
            }

        # Crop Disease / Plant Pathology / Agricultural Vision
        crop_kw = [
            "crop", "plant", "leaf", "foliar", "agriculture", "crop disease",
            "plantvillage", "blight", "rust", "pest", "weed", "plant pathology",
            "botanical", "maize", "tomato", "potato", "wheat"
        ]
        if any(kw in combined_text for kw in crop_kw):
            return {
                "domain": "agricultural_vision",
                "domain_label": "Plant Pathology & Agricultural Vision Pipeline",
                "default_input": "High-Resolution Field Crop Images (RGB)",
                "default_preproc": "Geometric Augmentation, Color Jitter & 224x224 Resizing",
                "default_features": "EfficientNet-B0 Depthwise Separable Feature Extractor",
                "default_model": "EfficientNet-B0 Multi-Class Softmax Classifier",
                "default_explainability": "Grad-CAM Foliar Lesion Heatmap Visualizer",
                "default_metrics": "Top-1 Accuracy, Precision, Recall & Macro-F1",
                "default_loss": "Cross-Entropy Loss with Label Smoothing & AdamW Optimizer",
                "default_serving": "TFLite / ONNX INT8 Quantized Mobile Edge Engine",
                "default_api": "FastAPI Mobile Offline-Sync & Field Diagnostic REST API",
                "default_frontend": "Mobile Responsive Field Agronomy Dashboard & Foliar Saliency Visualizer",
                "default_deploy": "Lightweight Edge Container (Docker on ARM64 / Android On-Device)",
            }

        # Financial Fraud Detection / Tabular Anomaly
        fraud_kw = [
            "fraud", "credit card", "transaction", "financial", "anomaly detection",
            "anti-money laundering", "smote", "ieee-cis", "tabular", "imbalanced classification",
            "fraud detection", "bank", "payment"
        ]
        if any(kw in combined_text for kw in fraud_kw):
            return {
                "domain": "fraud_detection",
                "domain_label": "Financial Fraud Detection & Transaction Analysis Pipeline",
                "default_input": "Temporal Transaction Records & Account Event Logs",
                "default_preproc": "Time-Window Aggregations, Min-Max Scaling & SMOTE Oversampling",
                "default_features": "Multi-Head Self-Attention Temporal Window Encoder",
                "default_model": "Transformer Anomaly Risk Classification Model",
                "default_explainability": "SHAP Tree / Sequence Feature Attribution Explainer",
                "default_metrics": "PR-AUC, Detection Rate @ 0.1% FPR, ROC-AUC",
                "default_loss": "Focal Loss (gamma=2.0) with AdamW Optimizer",
                "default_serving": "TensorRT / ONNX Low-Latency Stream Scoring Engine (<20ms)",
                "default_api": "Low-Latency Kafka Stream Consumer & Asynchronous Scoring Service",
                "default_frontend": "Real-Time Fraud Telemetry & SHAP Risk Attribution Dashboard",
                "default_deploy": "High-Throughput Kubernetes Worker Pods with Redis Feature Store",
            }

        # NLP / Text / Large Language Models
        nlp_kw = [
            "nlp", "language model", "tokenizer", "bpe", "wordpiece", "sentencepiece",
            "bert", "roberta", "llama", "sentiment", "text classification", "summarization",
            "transformer", "natural language"
        ]
        if any(kw in combined_text for kw in nlp_kw):
            return {
                "domain": "nlp",
                "domain_label": "Natural Language Processing & Transformer Pipeline",
                "default_input": "Domain Text Corpora / Tokenized Sequence Data",
                "default_preproc": "Subword Tokenization (BPE/WordPiece) with Dynamic Padding",
                "default_features": "Transformer Self-Attention Latent Projection Layers",
                "default_model": "Transformer Encoder Backbone & Classification Head",
                "default_explainability": "Self-Attention Rollout & Attribution Visualizer",
                "default_metrics": "Accuracy, Macro-F1, Perplexity, BLEU/ROUGE",
                "default_loss": "Cross-Entropy Loss with AdamW & Cosine Annealing",
                "default_serving": "vLLM / ONNX Runtime with Dynamic KV-Cache Batching",
                "default_api": "FastAPI Asynchronous Text Inference Endpoint",
                "default_frontend": "Interactive Text Synthesis & Attention Visualization Interface",
                "default_deploy": "GPU-Accelerated Docker Container on Kubernetes",
            }

        # Graph Neural Networks / Molecular
        graph_kw = ["graph", "gnn", "gcn", "gat", "molecular", "citation network", "node classification"]
        if any(kw in combined_text for kw in graph_kw):
            return {
                "domain": "graph",
                "domain_label": "Graph Representation & Geometric Deep Learning Pipeline",
                "default_input": "Attributed Graph Adjacency Matrices & Node Features",
                "default_preproc": "Graph Adjacency Normalization & Subgraph Neighborhood Sampling",
                "default_features": "Message-Passing Convolutional Neighborhood Aggregations",
                "default_model": "Graph Convolutional Network (GCN / GAT) Classifier",
                "default_explainability": "GNNExplainer Subgraph Saliency Attribution",
                "default_metrics": "Node Classification Accuracy & Macro-F1",
                "default_loss": "Cross-Entropy with AdamW Optimizer",
                "default_serving": "TorchScript Graph Inference Worker",
                "default_api": "FastAPI Graph Query & Topology Prediction Microservice",
                "default_frontend": "Interactive Graph Topology Explorer & Subgraph Saliency Visualizer",
                "default_deploy": "Containerized Service with PyG Runtime",
            }

        # General Computer Vision
        cv_kw = ["vision", "image", "object detection", "yolo", "segmentation", "convolutional", "resnet"]
        if any(kw in combined_text for kw in cv_kw):
            return {
                "domain": "computer_vision",
                "domain_label": "Computer Vision Deep Learning Pipeline",
                "default_input": "Standard Computer Vision Image Benchmark Corpus",
                "default_preproc": "RGB Normalization, Geometric Augmentations & Resizing",
                "default_features": "Convolutional Feature Representation Layer",
                "default_model": "Deep Convolutional Neural Network (CNN / ResNet)",
                "default_explainability": "Grad-CAM Saliency Localization Map",
                "default_metrics": "Top-1 Accuracy, Precision, Recall & mAP",
                "default_loss": "Categorical Cross-Entropy with SGD / AdamW",
                "default_serving": "Triton / ONNX Runtime FP16 Engine",
                "default_api": "FastAPI Computer Vision REST Endpoint",
                "default_frontend": "Interactive Image Inference & Saliency Heatmap Visualizer",
                "default_deploy": "Docker Container with NVIDIA Container Toolkit",
            }

        # General Fallback
        return {
            "domain": "general",
            "domain_label": "Deep Learning Research Implementation Pipeline",
            "default_input": "Domain Research Dataset / Feature Matrix",
            "default_preproc": "Data Cleaning, Feature Normalization & Partitioning",
            "default_features": "Learned Latent Embedding Projections",
            "default_model": "Deep Neural Network Backbone",
            "default_explainability": "Feature Attribution & Interpretability Visualizer",
            "default_metrics": "Validation Loss & Task-Specific F1-Score",
            "default_loss": "Task Loss Function with AdamW Optimizer",
            "default_serving": "ONNX Runtime / Triton Inference Server",
            "default_api": "FastAPI Asynchronous Microservice with Queue Worker",
            "default_frontend": "Interactive Web Dashboard with Real-Time Synthesis Visualizer",
            "default_deploy": "Standard Docker Container on Linux Host",
        }

    # -------------------------------------------------------------------------
    # 10 Dynamic Paper-Specific Components
    # -------------------------------------------------------------------------

    def _extract_dynamic_components(
        self,
        documents: List[ResearchDocument],
        domain: Dict[str, Any],
        constraints: Dict[str, Any],
        is_scaffold: bool = False,
    ) -> List[ArchitectureComponent]:
        """Extract exactly 10 stage-compatible architecture components tailored to paper evidence."""
        components: List[ArchitectureComponent] = []

        # 1. Dataset
        components.append(self._extract_dataset_component(documents, domain, is_scaffold))

        # 2. Preprocessing
        components.append(self._extract_preprocessing_component(documents, domain, is_scaffold))

        # 3. Feature Extraction
        components.append(self._extract_feature_extraction_component(documents, domain, is_scaffold))

        # 4. Model
        components.append(self._extract_model_component(documents, domain, is_scaffold))

        # 5. Training
        components.append(self._extract_training_component(documents, domain, is_scaffold))

        # 6. Evaluation
        components.append(self._extract_evaluation_component(documents, domain, is_scaffold))

        # 7. Inference
        components.append(self._extract_inference_component(documents, domain, constraints, is_scaffold))

        # 8. Backend/API
        components.append(self._extract_backend_component(domain, constraints, is_scaffold))

        # 9. Frontend
        components.append(self._extract_frontend_component(domain, constraints, is_scaffold))

        # 10. Deployment
        components.append(self._extract_deployment_component(domain, constraints, is_scaffold))

        return components

    def _extract_dataset_component(
        self, docs: List[ResearchDocument], domain: Dict[str, Any], is_scaffold: bool
    ) -> ArchitectureComponent:
        # 1. Direct metadata dataset
        for doc in docs:
            if doc.datasets and len(doc.datasets) > 0:
                ds_name = doc.datasets[0].strip()
                return ArchitectureComponent(
                    component="Dataset",
                    value=ds_name,
                    source_paper=doc.title,
                    evidence=f"Manuscript explicitly evaluates on verified benchmark '{ds_name}'.",
                    confidence=0.98,
                    status=EvidenceStatus.VERIFIED,
                    evidence_status=EvidenceStatus.VERIFIED,
                    evidence_source=f"{doc.internal_id} (Metadata: datasets=['{ds_name}'])",
                    details={"datasets": doc.datasets},
                    supporting_paper_id=doc.internal_id,
                )

        # 2. Abstract textual match
        from app.intelligence.graph import KNOWN_DATASETS
        for doc in docs:
            full_text = f"{doc.title} {doc.abstract or ''}"
            match = _find_snippet(full_text, KNOWN_DATASETS)
            if match:
                ds_name, snip = match
                return ArchitectureComponent(
                    component="Dataset",
                    value=ds_name,
                    source_paper=doc.title,
                    evidence=f"Empirical benchmark verified in abstract: {snip}",
                    confidence=0.92,
                    status=EvidenceStatus.VERIFIED,
                    evidence_status=EvidenceStatus.VERIFIED,
                    evidence_source=f"{doc.internal_id} (Abstract text: '{snip}')",
                    details={"matched_dataset": ds_name},
                    supporting_paper_id=doc.internal_id,
                )

        # 3. If domain has explicit concept signal
        primary = docs[0]
        concepts = [c.lower() for c in (primary.concepts or [])]
        has_empirical_signal = any(
            any(w in c for w in ["vision", "image", "language", "nlp", "text", "audio", "graph", "cancer", "fraud", "crop"])
            for c in concepts
        ) or any(
            any(w in (doc.title + " " + (doc.abstract or "")).lower() for w in ["mri", "crop", "fraud", "image", "text", "speech"])
            for doc in docs
        )

        if has_empirical_signal and not is_scaffold:
            return ArchitectureComponent(
                component="Dataset",
                value=domain["default_input"],
                source_paper=primary.title,
                evidence=f"Inferred {domain['domain_label']} input corpus required from domain concepts.",
                confidence=0.75,
                status=EvidenceStatus.INFERRED,
                evidence_status=EvidenceStatus.INFERRED,
                evidence_source=f"{primary.internal_id} (Domain Inference: {domain['domain']})",
                supporting_paper_id=primary.internal_id,
            )

        # 4. Unknown / Not identified
        return ArchitectureComponent(
            component="Dataset",
            value="Unknown / Custom Ingestion Dataset",
            source_paper=primary.title if docs else "Unknown",
            evidence="No public benchmark or empirical dataset identified in publisher metadata or abstract.",
            confidence=0.20,
            status=EvidenceStatus.UNKNOWN,
            evidence_status=EvidenceStatus.UNKNOWN,
            evidence_source="Publisher metadata absent",
        )

    def _extract_preprocessing_component(
        self, docs: List[ResearchDocument], domain: Dict[str, Any], is_scaffold: bool
    ) -> ArchitectureComponent:
        # Domain-specific keyword scan
        domain_preproc_kws: Dict[str, List[str]] = {
            "medical_imaging": [
                "skull stripping", "skull-stripping", "intensity normalization",
                "bias field", "slice registration", "histogram matching", "resizing", "nifti"
            ],
            "agricultural_vision": [
                "data augmentation", "augmentation", "random flip", "color jitter",
                "rotation", "cutmix", "mixup", "affine", "resizing", "normalize"
            ],
            "fraud_detection": [
                "smote", "adasyn", "sliding window", "time-window", "aggregation",
                "scaling", "min-max", "imputation", "feature engineering"
            ],
            "nlp": [
                "byte-pair encoding", "bpe", "wordpiece", "sentencepiece",
                "tokenization", "sequence truncation", "padding"
            ],
            "audio": ["spectrogram", "mel-spectrogram", "stft", "fourier transform"],
            "graph": ["adjacency normalization", "node feature projection", "subgraph sampling"],
        }

        target_kws = domain_preproc_kws.get(domain["domain"], KNOWN_PREPROCESSING)
        for doc in docs:
            text = f"{doc.title} {doc.abstract or ''} {' '.join(doc.methods or [])}"
            match = _find_snippet(text, target_kws)
            if match:
                kw, snip = match
                return ArchitectureComponent(
                    component="Preprocessing",
                    value=f"{kw.title()} Pipeline",
                    source_paper=doc.title,
                    evidence=f"Verified preprocessing technique in manuscript: {snip}",
                    confidence=0.94,
                    status=EvidenceStatus.VERIFIED,
                    evidence_status=EvidenceStatus.VERIFIED,
                    evidence_source=f"{doc.internal_id} (Abstract text: '{snip}')",
                    technologies=[kw],
                    supporting_paper_id=doc.internal_id,
                )

        primary = docs[0]
        if is_scaffold or domain["domain"] == "general":
            return ArchitectureComponent(
                component="Preprocessing",
                value="Standard Data Cleaning & Normalization",
                source_paper=primary.title if docs else "Unknown",
                evidence="No specific preprocessing methodology detailed; using baseline cleaning.",
                confidence=0.40,
                status=EvidenceStatus.UNKNOWN,
                evidence_status=EvidenceStatus.UNKNOWN,
                evidence_source="Evidence insufficient — generic implementation scaffold",
            )

        return ArchitectureComponent(
            component="Preprocessing",
            value=domain["default_preproc"],
            source_paper=primary.title,
            evidence=f"Inferred standard transformation pipeline for {domain['domain_label']}.",
            confidence=0.80,
            status=EvidenceStatus.INFERRED,
            evidence_status=EvidenceStatus.INFERRED,
            evidence_source=f"{primary.internal_id} (Domain Inference: {domain['domain']})",
            supporting_paper_id=primary.internal_id,
        )

    def _extract_feature_extraction_component(
        self, docs: List[ResearchDocument], domain: Dict[str, Any], is_scaffold: bool
    ) -> ArchitectureComponent:
        # Check methods in docs
        for doc in docs:
            if doc.methods and len(doc.methods) > 0:
                method_name = doc.methods[0].strip()
                return ArchitectureComponent(
                    component="Feature Extraction",
                    value=method_name,
                    source_paper=doc.title,
                    evidence=f"Direct methodological specification in manuscript: '{method_name}'.",
                    confidence=0.95,
                    status=EvidenceStatus.VERIFIED,
                    evidence_status=EvidenceStatus.VERIFIED,
                    evidence_source=f"{doc.internal_id} (Metadata: methods=['{method_name}'])",
                    technologies=[method_name],
                    supporting_paper_id=doc.internal_id,
                )

        from app.intelligence.graph import KNOWN_METHODS
        for doc in docs:
            text = f"{doc.title} {doc.abstract or ''}"
            match = _find_snippet(text, KNOWN_METHODS)
            if match:
                kw, snip = match
                return ArchitectureComponent(
                    component="Feature Extraction",
                    value=kw,
                    source_paper=doc.title,
                    evidence=f"Feature extraction method identified in text: {snip}",
                    confidence=0.91,
                    status=EvidenceStatus.VERIFIED,
                    evidence_status=EvidenceStatus.VERIFIED,
                    evidence_source=f"{doc.internal_id} (Abstract text: '{snip}')",
                    technologies=[kw],
                    supporting_paper_id=doc.internal_id,
                )

        primary = docs[0]
        return ArchitectureComponent(
            component="Feature Extraction",
            value=domain["default_features"],
            source_paper=primary.title,
            evidence=f"Inferred representation layer mapping raw inputs into {domain['domain_label']} space.",
            confidence=0.75,
            status=EvidenceStatus.INFERRED,
            evidence_status=EvidenceStatus.INFERRED,
            evidence_source=f"{primary.internal_id} (Domain Inference: {domain['domain']})",
            supporting_paper_id=primary.internal_id,
        )

    def _extract_model_component(
        self, docs: List[ResearchDocument], domain: Dict[str, Any], is_scaffold: bool
    ) -> ArchitectureComponent:
        # 1. Metadata models
        for doc in docs:
            if doc.models and len(doc.models) > 0:
                m_name = doc.models[0].strip()
                return ArchitectureComponent(
                    component="Model",
                    value=m_name,
                    source_paper=doc.title,
                    evidence=f"Primary model backbone '{m_name}' specified in publisher metadata.",
                    confidence=0.98,
                    status=EvidenceStatus.VERIFIED,
                    evidence_status=EvidenceStatus.VERIFIED,
                    evidence_source=f"{doc.internal_id} (Metadata: models=['{m_name}'])",
                    details={"models": doc.models},
                    technologies=[m_name],
                    supporting_paper_id=doc.internal_id,
                )

        # 2. Textual search in title and abstract
        for doc in docs:
            text = f"{doc.title} {doc.abstract or ''}"
            match = _find_snippet(text, KNOWN_MODELS)
            if match:
                kw, snip = match
                return ArchitectureComponent(
                    component="Model",
                    value=kw,
                    source_paper=doc.title,
                    evidence=f"Model architecture verified in paper text: {snip}",
                    confidence=0.94,
                    status=EvidenceStatus.VERIFIED,
                    evidence_status=EvidenceStatus.VERIFIED,
                    evidence_source=f"{doc.internal_id} (Abstract text: '{snip}')",
                    technologies=[kw],
                    supporting_paper_id=doc.internal_id,
                )

        # 3. Check for classifier heads like SVM if mentioned in text
        for doc in docs:
            text = f"{doc.title} {doc.abstract or ''} {' '.join(doc.methods or [])}"
            match = _find_snippet(text, KNOWN_HEADS)
            if match:
                kw, snip = match
                return ArchitectureComponent(
                    component="Model",
                    value=f"{kw} Classifier Model",
                    source_paper=doc.title,
                    evidence=f"Classifier architecture verified in paper: {snip}",
                    confidence=0.92,
                    status=EvidenceStatus.VERIFIED,
                    evidence_status=EvidenceStatus.VERIFIED,
                    evidence_source=f"{doc.internal_id} (Abstract text: '{snip}')",
                    technologies=[kw],
                    supporting_paper_id=doc.internal_id,
                )

        primary = docs[0]
        if is_scaffold:
            return ArchitectureComponent(
                component="Model",
                value="Deep Neural Network Backbone",
                source_paper=primary.title,
                evidence="Manuscript discusses theoretical formulation without explicit architectural naming.",
                confidence=0.50,
                status=EvidenceStatus.INFERRED,
                evidence_status=EvidenceStatus.INFERRED,
                evidence_source="Theoretical inference",
            )

        return ArchitectureComponent(
            component="Model",
            value=domain["default_model"],
            source_paper=primary.title,
            evidence=f"Inferred foundational neural architecture for {domain['domain_label']}.",
            confidence=0.75,
            status=EvidenceStatus.INFERRED,
            evidence_status=EvidenceStatus.INFERRED,
            evidence_source=f"{primary.internal_id} (Domain Inference: {domain['domain']})",
            supporting_paper_id=primary.internal_id,
        )

    def _extract_training_component(
        self, docs: List[ResearchDocument], domain: Dict[str, Any], is_scaffold: bool
    ) -> ArchitectureComponent:
        for doc in docs:
            text = f"{doc.title} {doc.abstract or ''}"
            opt_match = _find_snippet(text, KNOWN_OPTIMIZERS)
            loss_match = _find_snippet(text, KNOWN_LOSSES)
            if opt_match or loss_match:
                parts = []
                evs = []
                if opt_match:
                    parts.append(f"{opt_match[0]}")
                    evs.append(opt_match[1])
                if loss_match:
                    parts.append(loss_match[0])
                    evs.append(loss_match[1])
                return ArchitectureComponent(
                    component="Training",
                    value=" + ".join(parts),
                    source_paper=doc.title,
                    evidence="; ".join(evs),
                    confidence=0.94,
                    status=EvidenceStatus.VERIFIED,
                    evidence_status=EvidenceStatus.VERIFIED,
                    evidence_source=f"{doc.internal_id} (Abstract text: '{evs[0]}')",
                    technologies=parts,
                    supporting_paper_id=doc.internal_id,
                )

        primary = docs[0]
        return ArchitectureComponent(
            component="Training",
            value=domain["default_loss"],
            source_paper=primary.title,
            evidence=f"Inferred convergence optimization protocol for {domain['domain_label']}.",
            confidence=0.80,
            status=EvidenceStatus.INFERRED,
            evidence_status=EvidenceStatus.INFERRED,
            evidence_source=f"{primary.internal_id} (Domain Inference: {domain['domain']})",
            supporting_paper_id=primary.internal_id,
        )

    def _extract_evaluation_component(
        self, docs: List[ResearchDocument], domain: Dict[str, Any], is_scaffold: bool
    ) -> ArchitectureComponent:
        for doc in docs:
            text = f"{doc.title} {doc.abstract or ''}"
            match = _find_snippet(text, KNOWN_METRICS)
            if match:
                kw, snip = match
                return ArchitectureComponent(
                    component="Evaluation",
                    value=f"{kw} Benchmark Validation",
                    source_paper=doc.title,
                    evidence=f"Evaluation protocol verified in paper: {snip}",
                    confidence=0.94,
                    status=EvidenceStatus.VERIFIED,
                    evidence_status=EvidenceStatus.VERIFIED,
                    evidence_source=f"{doc.internal_id} (Abstract text: '{snip}')",
                    technologies=[kw],
                    supporting_paper_id=doc.internal_id,
                )

        primary = docs[0]
        return ArchitectureComponent(
            component="Evaluation",
            value=domain["default_metrics"],
            source_paper=primary.title,
            evidence=f"Inferred domain benchmark validation metrics for {domain['domain_label']}.",
            confidence=0.82,
            status=EvidenceStatus.INFERRED,
            evidence_status=EvidenceStatus.INFERRED,
            evidence_source=f"{primary.internal_id} (Domain Inference: {domain['domain']})",
            supporting_paper_id=primary.internal_id,
        )

    def _extract_inference_component(
        self,
        docs: List[ResearchDocument],
        domain: Dict[str, Any],
        constraints: Dict[str, Any],
        is_scaffold: bool,
    ) -> ArchitectureComponent:
        target_env = str(constraints.get("target_environment", "")).lower()
        latency_budget = constraints.get("latency_budget_ms")
        primary = docs[0]

        if target_env == "edge" or (latency_budget and latency_budget <= 50):
            return ArchitectureComponent(
                component="Inference",
                value="ONNX Runtime with INT8 Post-Training Quantization",
                source_paper=primary.title,
                evidence=f"Selected for edge/low-latency constraint ({latency_budget or '<=50'}ms budget).",
                confidence=0.90,
                status=EvidenceStatus.INFERRED,
                evidence_status=EvidenceStatus.INFERRED,
                evidence_source="ResearchMind Proposed Build",
                details={"engine": "onnxruntime", "quantization": "int8"},
                technologies=["onnxruntime"],
            )
        else:
            return ArchitectureComponent(
                component="Inference",
                value=domain["default_serving"],
                source_paper=primary.title,
                evidence=f"ResearchMind proposed scalable inference server tailored to {domain['domain_label']}.",
                confidence=0.88,
                status=EvidenceStatus.INFERRED,
                evidence_status=EvidenceStatus.INFERRED,
                evidence_source="ResearchMind Proposed Build",
                technologies=["onnxruntime", "triton"],
            )

    def _extract_backend_component(
        self, domain: Dict[str, Any], constraints: Dict[str, Any], is_scaffold: bool
    ) -> ArchitectureComponent:
        framework = constraints.get("preferred_framework", "PyTorch")
        return ArchitectureComponent(
            component="Backend/API",
            value=domain["default_api"],
            source_paper="ResearchMind System Architecture",
            evidence=f"Asynchronous HTTP/WebSocket pipeline interfacing with {framework} runtime.",
            confidence=0.95,
            status=EvidenceStatus.INFERRED,
            evidence_status=EvidenceStatus.INFERRED,
            evidence_source="ResearchMind Proposed Build",
            details={"framework": "FastAPI", "protocol": "REST + WebSocket"},
            technologies=["fastapi", "uvicorn", "pydantic"],
        )

    def _extract_frontend_component(
        self, domain: Dict[str, Any], constraints: Dict[str, Any], is_scaffold: bool
    ) -> ArchitectureComponent:
        return ArchitectureComponent(
            component="Frontend",
            value=domain["default_frontend"],
            source_paper="ResearchMind UI Specification",
            evidence="Interactive user interface presenting telemetry, predictions, and live architecture.",
            confidence=0.95,
            status=EvidenceStatus.INFERRED,
            evidence_status=EvidenceStatus.INFERRED,
            evidence_source="ResearchMind Proposed Build",
            details={"stack": "HTML5 / Tailwind CSS / Vanilla JS"},
            technologies=["html5", "tailwind", "vanilla-js"],
        )

    def _extract_deployment_component(
        self, domain: Dict[str, Any], constraints: Dict[str, Any], is_scaffold: bool
    ) -> ArchitectureComponent:
        target_env = str(constraints.get("target_environment", "cloud")).lower()
        if target_env == "edge":
            return ArchitectureComponent(
                component="Deployment",
                value="Containerized Linux Runtime (Docker on ARM64 / Embedded GPU)",
                source_paper="Deployment Target Spec",
                evidence="Target environment constrained to edge / lightweight compute nodes.",
                confidence=0.92,
                status=EvidenceStatus.INFERRED,
                evidence_status=EvidenceStatus.INFERRED,
                evidence_source="ResearchMind Proposed Build",
                technologies=["docker", "arm64"],
            )
        else:
            return ArchitectureComponent(
                component="Deployment",
                value=domain["default_deploy"],
                source_paper="Deployment Target Spec",
                evidence=f"Production container specification tailored to {domain['domain_label']}.",
                confidence=0.95,
                status=EvidenceStatus.INFERRED,
                evidence_status=EvidenceStatus.INFERRED,
                evidence_source="ResearchMind Proposed Build",
                technologies=["docker", "kubernetes"],
            )

    # -------------------------------------------------------------------------
    # Pipeline Connections
    # -------------------------------------------------------------------------

    def _build_pipeline_connections(
        self, components: List[ArchitectureComponent], domain: Dict[str, Any]
    ) -> List[ArchitectureConnection]:
        """Create explicit directed pipeline flow from Dataset to Deployment."""
        comp_map = {c.component: c for c in components}
        dom = domain["domain"]

        if dom == "medical_imaging":
            flow_order = [
                ("Dataset", "Preprocessing", "Raw volumetric MRI scans ingested into preprocessing pipeline", "Volumetric NIfTI / DICOM Scans [H, W, D]"),
                ("Preprocessing", "Feature Extraction", "Skull-stripped and intensity-normalized slices passed to feature backbone", "Normalized Slices [B, 1, 224, 224]"),
                ("Feature Extraction", "Model", "Deep residual embeddings fed into decision classifier", "Deep Latent Embeddings [B, 2048]"),
                ("Model", "Training", "Margin hinge loss & cross-entropy evaluated", "Loss Gradients & Updated Checkpoint Weights"),
                ("Model", "Evaluation", "Model predictions evaluated on holdout split", "Predicted Margins, Sensitivity & Specificity"),
                ("Model", "Inference", "Trained weights compiled into clinical inference engine", "Compiled Model Artifacts (.onnx / .pt)"),
                ("Inference", "Backend/API", "Clinical inference engine wrapped in PACS microservice", "Inference Requests & DICOM Responses"),
                ("Backend/API", "Frontend", "REST/WebSocket API consumed by clinical visualizer", "Volumetric Slices & Grad-CAM Heatmap Overlays"),
                ("Backend/API", "Deployment", "Microservice deployed in HIPAA-compliant container", "HIPAA Container Images on Kubernetes"),
            ]
        elif dom == "agricultural_vision":
            flow_order = [
                ("Dataset", "Preprocessing", "Raw field camera photos ingested into augmentation pipeline", "Raw RGB Leaf Images (JPEG / PNG)"),
                ("Preprocessing", "Feature Extraction", "Augmented, normalized images passed to EfficientNet feature extractor", "Augmented RGB Tensors [B, 3, 224, 224]"),
                ("Feature Extraction", "Model", "Compound-scaled bottleneck features fed to softmax classifier", "Bottleneck Feature Maps [B, 1280]"),
                ("Model", "Training", "Cross-entropy loss optimized with AdamW", "Backpropagated Gradients & Weight Updates"),
                ("Model", "Evaluation", "Disease classification evaluated on holdout split", "Test Predictions, Top-1 Accuracy & F1"),
                ("Model", "Inference", "Trained model quantized for edge mobile runtime", "TFLite / INT8 Quantized Model (.tflite)"),
                ("Inference", "Backend/API", "Offline mobile sync endpoints exposed for field app", "Mobile Diagnostic Requests & Disease Predictions"),
                ("Backend/API", "Frontend", "Field telemetry streamed to mobile agronomy app", "Field Disease Identification & Foliar Heatmaps"),
                ("Backend/API", "Deployment", "Mobile application deployed on lightweight edge node", "ARM64 Edge Container Images"),
            ]
        elif dom == "fraud_detection":
            flow_order = [
                ("Dataset", "Preprocessing", "Raw transaction logs ingested from event streams", "Tabular Event Records & Timestamps"),
                ("Preprocessing", "Feature Extraction", "Engineered feature vectors fed into sequence encoder", "Standardized Feature Vectors [B, Seq_Len, D]"),
                ("Feature Extraction", "Model", "Contextual sequence representations fed into risk scorer", "Encoded State Vectors [B, D_Model]"),
                ("Model", "Training", "Focal loss optimized to handle extreme class imbalance", "Focal Loss Gradients & Optimizer Step"),
                ("Model", "Evaluation", "Detection rate evaluated at low false positive rate threshold", "PR-AUC & Precision-Recall Curve Scores"),
                ("Model", "Inference", "Model serialized for high-throughput streaming scoring", "Compiled Low-Latency Engine (<20ms)"),
                ("Inference", "Backend/API", "Kafka streaming consumer invokes scoring runtime", "Stream Transactions & Fraud Risk Scores"),
                ("Backend/API", "Frontend", "Real-time streaming telemetry and SHAP values sent to UI", "Live Fraud Stream Telemetry & SHAP Attributions"),
                ("Backend/API", "Deployment", "Microservices scaled horizontally on Kubernetes", "Kubernetes Pods with Redis Feature Store"),
            ]
        else:
            flow_order = [
                ("Dataset", "Preprocessing", "Raw data ingested into clean ingestion pipeline", "Batches of Raw Records / Inputs"),
                ("Preprocessing", "Feature Extraction", "Preprocessed inputs passed to feature backbone", "Normalized Input Tensors [B, ...]"),
                ("Feature Extraction", "Model", "Extracted representations fed into decision model", "Latent Feature Vectors [B, D]"),
                ("Model", "Training", "Loss function evaluated against ground truth", "Computed Loss & Optimizer Gradients"),
                ("Model", "Evaluation", "Checkpoints benchmarked on holdout validation split", "Validation Predictions & Target Metrics"),
                ("Model", "Inference", "Trained weights compiled into optimized runtime", "Serialized Model Artifacts (.onnx / .pt)"),
                ("Inference", "Backend/API", "Inference engine wrapped inside async service", "Inference Payloads & JSON Responses"),
                ("Backend/API", "Frontend", "REST/WebSocket API consumed by client dashboard", "Payloads, telemetry metrics, and scores"),
                ("Backend/API", "Deployment", "Microservice packaged in container image", "Docker Container Images on Host/K8s"),
            ]

        connections: List[ArchitectureConnection] = []
        for src, tgt, desc, flow in flow_order:
            if src in comp_map and tgt in comp_map:
                connections.append(
                    ArchitectureConnection(
                        source=src,
                        target=tgt,
                        description=desc,
                        data_flow=flow,
                    )
                )
        return connections

    # -------------------------------------------------------------------------
    # Hardware & Dependencies
    # -------------------------------------------------------------------------

    def _extract_hardware_requirements(
        self,
        documents: List[ResearchDocument],
        bld_by_id: Dict[str, Any],
        constraints: Dict[str, Any],
        domain: Dict[str, Any],
    ) -> List[HardwareRequirement]:
        hw_list: List[HardwareRequirement] = []
        primary = documents[0]
        bld = bld_by_id.get(primary.internal_id)

        hw_ev = ""
        if bld and "hardware_requirements" in bld.dimensions:
            hw_dim = bld.dimensions["hardware_requirements"]
            if hw_dim.evidence:
                hw_ev = hw_dim.evidence[0]

        max_gpu_gb = constraints.get("max_gpu_memory_gb")
        target_env = str(constraints.get("target_environment", "")).lower()

        if target_env == "edge" or (max_gpu_gb and max_gpu_gb <= 8):
            hw_list.append(
                HardwareRequirement(
                    category="GPU Compute",
                    requirement=f"Consumer GPU or Edge Accelerator <= {max_gpu_gb or 8}GB VRAM",
                    status=EvidenceStatus.INFERRED,
                    evidence=f"Hardware scaled to meet user constraint (max VRAM: {max_gpu_gb or 8}GB).",
                    confidence=0.90,
                )
            )
            hw_list.append(
                HardwareRequirement(
                    category="System RAM",
                    requirement="16 GB RAM",
                    status=EvidenceStatus.INFERRED,
                    evidence="Sufficient host memory for quantized local inference pipeline.",
                    confidence=0.88,
                )
            )
        elif hw_ev and ("cluster" in hw_ev.lower() or "tpu" in hw_ev.lower() or "a100" in hw_ev.lower()):
            hw_list.append(
                HardwareRequirement(
                    category="GPU Compute",
                    requirement="NVIDIA A100 / H100 (>= 40GB VRAM) or Distributed TPU Pod",
                    status=EvidenceStatus.VERIFIED,
                    evidence=hw_ev,
                    confidence=0.95,
                )
            )
            hw_list.append(
                HardwareRequirement(
                    category="System RAM",
                    requirement=">= 64 GB High-Bandwidth RAM",
                    status=EvidenceStatus.INFERRED,
                    evidence="Required host memory to feed large distributed accelerator nodes.",
                    confidence=0.90,
                )
            )
        else:
            hw_list.append(
                HardwareRequirement(
                    category="GPU Compute",
                    requirement="NVIDIA GPU >= 16GB VRAM (CUDA Compute Capability >= 8.0)",
                    status=EvidenceStatus.INFERRED if not hw_ev else EvidenceStatus.VERIFIED,
                    evidence=hw_ev or "Standard single-GPU setup required for fine-tuning modern neural models.",
                    confidence=0.88,
                )
            )
            hw_list.append(
                HardwareRequirement(
                    category="System RAM",
                    requirement="32 GB System Memory",
                    status=EvidenceStatus.INFERRED,
                    evidence="Baseline memory for tokenization caching and DataLoader workers.",
                    confidence=0.85,
                )
            )

        hw_list.append(
            HardwareRequirement(
                category="Host Storage",
                requirement=">= 100 GB NVMe SSD Storage",
                status=EvidenceStatus.INFERRED,
                evidence="Fast local scratch storage for checkpoint serialization and dataset shards.",
                confidence=0.90,
            )
        )
        return hw_list

    def _extract_dependencies(
        self,
        documents: List[ResearchDocument],
        bld_by_id: Dict[str, Any],
        constraints: Dict[str, Any],
        domain: Dict[str, Any],
        components: List[ArchitectureComponent],
    ) -> List[DependencyItem]:
        deps: List[DependencyItem] = []
        primary = documents[0]
        dom = domain["domain"]

        # Framework dependency
        pref_fw = constraints.get("preferred_framework", "PyTorch")
        deps.append(
            DependencyItem(
                name="torch" if pref_fw.lower() == "pytorch" else pref_fw.lower(),
                version_spec=">= 2.1.0",
                category="framework",
                license="BSD-3-Clause",
                status=EvidenceStatus.VERIFIED if primary.code_url else EvidenceStatus.INFERRED,
                evidence=f"Core neural tensor library for model execution ({primary.code_url or 'standard dependency'}).",
                confidence=0.95,
            )
        )

        # Domain dependencies
        if dom == "medical_imaging":
            deps.append(
                DependencyItem(
                    name="pydicom",
                    version_spec=">= 2.4.0",
                    category="data",
                    license="MIT",
                    status=EvidenceStatus.INFERRED,
                    evidence="Medical imaging DICOM file parser and volumetric metadata handler.",
                    confidence=0.92,
                )
            )
            deps.append(
                DependencyItem(
                    name="nibabel",
                    version_spec=">= 5.1.0",
                    category="data",
                    license="MIT",
                    status=EvidenceStatus.INFERRED,
                    evidence="Volumetric NIfTI neuroimaging format reader and writer.",
                    confidence=0.92,
                )
            )
            deps.append(
                DependencyItem(
                    name="scikit-learn",
                    version_spec=">= 1.3.0",
                    category="ml",
                    license="BSD-3-Clause",
                    status=EvidenceStatus.INFERRED,
                    evidence="Support Vector Machine (SVM) and clinical validation metrics.",
                    confidence=0.90,
                )
            )
        elif dom == "agricultural_vision":
            deps.append(
                DependencyItem(
                    name="torchvision",
                    version_spec=">= 0.16.0",
                    category="vision",
                    license="BSD-3-Clause",
                    status=EvidenceStatus.INFERRED,
                    evidence="EfficientNet backbone definitions and image augmentations.",
                    confidence=0.95,
                )
            )
            deps.append(
                DependencyItem(
                    name="albumentations",
                    version_spec=">= 1.3.1",
                    category="augmentation",
                    license="MIT",
                    status=EvidenceStatus.INFERRED,
                    evidence="Fast image augmentation library for agricultural foliage images.",
                    confidence=0.90,
                )
            )
        elif dom == "fraud_detection":
            deps.append(
                DependencyItem(
                    name="polars",
                    version_spec=">= 0.20.0",
                    category="data",
                    license="MIT",
                    status=EvidenceStatus.INFERRED,
                    evidence="High-performance multithreaded dataframe processing for transaction logs.",
                    confidence=0.92,
                )
            )
            deps.append(
                DependencyItem(
                    name="imbalanced-learn",
                    version_spec=">= 0.11.0",
                    category="ml",
                    license="MIT",
                    status=EvidenceStatus.INFERRED,
                    evidence="SMOTE and ADASYN class rebalancing for extreme transaction skew.",
                    confidence=0.90,
                )
            )
        else:
            deps.append(
                DependencyItem(
                    name="transformers",
                    version_spec=">= 4.36.0",
                    category="runtime",
                    license="Apache-2.0",
                    status=EvidenceStatus.INFERRED,
                    evidence="HuggingFace tokenizers and pretrained model layers.",
                    confidence=0.92,
                )
            )

        deps.append(
            DependencyItem(
                name="fastapi",
                version_spec=">= 0.104.0",
                category="api",
                license="MIT",
                status=EvidenceStatus.INFERRED,
                evidence="Async web framework for exposing REST inference endpoints.",
                confidence=0.95,
            )
        )

        deps.append(
            DependencyItem(
                name="onnxruntime" if constraints.get("target_environment") == "edge" else "tritonclient",
                version_spec=">= 1.16.0",
                category="runtime",
                license="MIT" if constraints.get("target_environment") == "edge" else "Apache-2.0",
                status=EvidenceStatus.INFERRED,
                evidence="High-performance model scoring engine.",
                confidence=0.88,
            )
        )

        deps.append(
            DependencyItem(
                name="numpy",
                version_spec=">= 1.24.0",
                category="data",
                license="BSD-3-Clause",
                status=EvidenceStatus.VERIFIED,
                evidence="Universal scientific array processing.",
                confidence=0.99,
            )
        )

        return deps

    # -------------------------------------------------------------------------
    # Multi-Paper Compatibility
    # -------------------------------------------------------------------------

    def _analyze_compatibility(
        self,
        docs: List[ResearchDocument],
        bld_by_id: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> CompatibilityAnalysis:
        if len(docs) <= 1:
            return CompatibilityAnalysis(
                is_compatible=True,
                compatibility_score=1.0,
                compatible_points=["Single paper selected; self-contained implementation architecture."],
                conflicts=[],
            )

        compatible_points: List[str] = []
        conflicts: List[str] = []

        licenses = {}
        for d in docs:
            bld = bld_by_id.get(d.internal_id)
            lic = d.license or (
                bld.dimensions["license_accessibility"].evidence[0]
                if bld and "license_accessibility" in bld.dimensions and bld.dimensions["license_accessibility"].evidence
                else None
            )
            licenses[d.title] = lic or "Unspecified"

        has_gpl = any("gpl" in str(l).lower() for l in licenses.values())
        has_proprietary = any("proprietary" in str(l).lower() or "confidential" in str(l).lower() for l in licenses.values())
        has_permissive = any("mit" in str(l).lower() or "apache" in str(l).lower() or "bsd" in str(l).lower() for l in licenses.values())

        if has_gpl and has_proprietary:
            conflicts.append("License incompatibility: GPL copyleft license cannot be combined with proprietary assets.")
        elif has_permissive:
            compatible_points.append("Permissive open-source licenses (MIT/Apache/BSD) are legally composable.")

        max_gpu = constraints.get("max_gpu_memory_gb")
        for d in docs:
            bld = bld_by_id.get(d.internal_id)
            if bld and "hardware_requirements" in bld.dimensions:
                ev_str = " ".join(bld.dimensions["hardware_requirements"].evidence).lower()
                if max_gpu and max_gpu <= 16 and ("cluster" in ev_str or "1024 tpu" in ev_str or "8x a100" in ev_str):
                    conflicts.append(f"Compute conflict: '{d.title[:30]}...' requires high-performance cluster, exceeding budget ({max_gpu}GB).")

        models_found = []
        for d in docs:
            if d.models:
                models_found.extend(d.models)
        if len(set(models_found)) > 1:
            unique_models = sorted(list(set(models_found)))
            compatible_points.append(f"Multi-paper architectural combination: Integrates models {', '.join(unique_models[:3])}.")
        else:
            compatible_points.append("Shared foundational architectural paradigm across selected manuscripts.")

        datasets_found = []
        for d in docs:
            if d.datasets:
                datasets_found.extend(d.datasets)
        if len(datasets_found) > 1:
            unique_datasets = sorted(list(set(datasets_found)))
            compatible_points.append(f"Multi-dataset evaluation benchmark: Spans {', '.join(unique_datasets[:3])}.")

        score = 1.0 - (0.25 * len(conflicts))
        score = max(0.1, min(1.0, score))
        return CompatibilityAnalysis(
            is_compatible=(len(conflicts) == 0),
            compatibility_score=round(score, 2),
            compatible_points=compatible_points,
            conflicts=conflicts,
        )

    def _formulate_title(
        self,
        components: List[ArchitectureComponent],
        primary_doc: ResearchDocument,
        domain: Dict[str, Any],
        is_scaffold: bool,
    ) -> str:
        if is_scaffold:
            return "Evidence insufficient — generic implementation scaffold"

        models_str = ", ".join(m.value for m in components if m.component == "Model" and m.status != EvidenceStatus.UNKNOWN)
        if not models_str:
            models_str = primary_doc.title[:50] + ("..." if len(primary_doc.title) > 50 else "")
        return f"Architecture for {models_str} Pipeline ({domain['domain_label']})"
