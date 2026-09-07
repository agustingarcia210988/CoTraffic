#!/usr/bin/env python3
"""
Market Basket Analysis & Semantic Clustering for CoTraffic.
Identifies complementary startup groups for cross-promotional campaigns.
"""

import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_CLUSTER_SIZE = 4
MAX_CLUSTER_SIZE = 5
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 2

@dataclass
class Startup:
    """Represents a startup/app with core metadata."""
    id: str
    nombre: str
    categoria: str
    descripcion: str
    tags: List[str]
    audiencia_objetivo: str
    url: Optional[str] = None
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data.pop('embedding', None)
        return data

@dataclass
class CoMarketingCluster:
    """Represents a cluster of complementary startups for co-marketing."""
    cluster_id: str
    audiencia_objetivo: str
    startups: List[Dict]
    complementarity_score: float
    rationale: str

    def to_dict(self) -> Dict:
        return {
            'cluster_id': self.cluster_id,
            'audiencia_objetivo': self.audiencia_objetivo,
            'startups': self.startups,
            'complementarity_score': round(self.complementarity_score, 4),
            'rationale': self.rationale,
            'size': len(self.startups)
        }

def load_startup_data(data_source: Optional[str] = None) -> List[Startup]:
    """Load startup data from JSON file or use built-in example data."""
    if data_source and Path(data_source).exists():
        logger.info(f"Loading startup data from {data_source}")
        try:
            with open(data_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [Startup(**item) for item in data]
        except Exception as e:
            logger.error(f"Error loading data from {data_source}: {e}")
            logger.info("Falling back to example data")

    example_data = [
        Startup(id="1", nombre="TaskFlow", categoria="Productividad",
            descripcion="Herramienta de gestión de tareas colaborativa para equipos remotos",
            tags=["tareas", "colaboración", "equipos"],
            audiencia_objetivo="Freelancers & Remote Teams",
            url="https://taskflow.example.com"),
        Startup(id="2", nombre="TimeLogger", categoria="Tracking",
            descripcion="Software de time tracking automatizado para freelancers",
            tags=["time-tracking", "facturación", "freelance"],
            audiencia_objetivo="Freelancers & Remote Teams",
            url="https://timelogger.example.com"),
        Startup(id="3", nombre="ProposalMaker", categoria="Ventas",
            descripcion="Generador de propuestas profesionales con plantillas",
            tags=["propuestas", "B2B", "ventas"],
            audiencia_objetivo="Freelancers & Remote Teams",
            url="https://proposalmaker.example.com"),
        Startup(id="4", nombre="ClientPortal", categoria="CRM",
            descripcion="Portal seguro para comunicación cliente-proveedor",
            tags=["comunicación", "cliente", "gestión"],
            audiencia_objetivo="Freelancers & Remote Teams",
            url="https://clientportal.example.com"),
        Startup(id="5", nombre="ShopAI", categoria="E-commerce",
            descripcion="Plataforma de e-commerce con IA para recomendaciones de productos",
            tags=["e-commerce", "IA", "recomendaciones"],
            audiencia_objetivo="E-commerce Stores",
            url="https://shopai.example.com"),
        Startup(id="6", nombre="InvTracker", categoria="Inventory",
            descripcion="Sistema de gestión de inventario en tiempo real",
            tags=["inventario", "stock", "e-commerce"],
            audiencia_objetivo="E-commerce Stores",
            url="https://invtracker.example.com"),
        Startup(id="7", nombre="ReviewBooster", categoria="Marketing",
            descripcion="Herramienta para recopilar y mostrar reseñas de clientes",
            tags=["reviews", "social-proof", "marketing"],
            audiencia_objetivo="E-commerce Stores",
            url="https://reviewbooster.example.com"),
        Startup(id="8", nombre="ShippingPro", categoria="Logística",
            descripcion="Optimización de envíos y cálculo de costos multicarrier",
            tags=["envíos", "logística", "e-commerce"],
            audiencia_objetivo="E-commerce Stores",
            url="https://shippingpro.example.com"),
        Startup(id="9", nombre="CodeDeploy", categoria="DevOps",
            descripcion="Pipeline CI/CD simplificado para desarrolladores",
            tags=["CI/CD", "deployment", "DevOps"],
            audiencia_objetivo="Developers & Dev Teams",
            url="https://codedeploy.example.com"),
        Startup(id="10", nombre="BugTracker", categoria="QA",
            descripcion="Sistema de seguimiento de errores integrado",
            tags=["bugs", "testing", "QA"],
            audiencia_objetivo="Developers & Dev Teams",
            url="https://bugtracker.example.com"),
        Startup(id="11", nombre="DocHub", categoria="Documentación",
            descripcion="Plataforma colaborativa para documentación de APIs",
            tags=["documentación", "API", "colaboración"],
            audiencia_objetivo="Developers & Dev Teams",
            url="https://dochub.example.com"),
        Startup(id="12", nombre="MonitorPro", categoria="Observability",
            descripcion="Monitoreo de aplicaciones y análisis de performance",
            tags=["monitoreo", "performance", "observability"],
            audiencia_objetivo="Developers & Dev Teams",
            url="https://monitorpro.example.com"),
    ]
    logger.info(f"Loaded {len(example_data)} example startups")
    return example_data

def generate_embeddings(startups: List[Startup], model_name: str = "all-MiniLM-L6-v2") -> List[Startup]:
    """Generate semantic embeddings for startup descriptions."""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = [f"{s.descripcion}. Tags: {', '.join(s.tags)}" for s in startups]
    logger.info(f"Generating embeddings for {len(startups)} startups...")
    embeddings = model.encode(texts, show_progress_bar=True)
    for startup, embedding in zip(startups, embeddings):
        startup.embedding = embedding
    logger.info("Embeddings generated successfully")
    return startups

def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

def cluster_by_audience(startups: List[Startup]) -> Dict[str, List[Startup]]:
    """Group startups by shared target audience."""
    audience_groups = {}
    for startup in startups:
        audience = startup.audiencia_objetivo
        if audience not in audience_groups:
            audience_groups[audience] = []
        audience_groups[audience].append(startup)
    logger.info(f"Found {len(audience_groups)} distinct audiences")
    return audience_groups

def is_valid_cluster(startups: List[Startup]) -> bool:
    """Check if cluster respects 'no duplicate categories' constraint."""
    categories = [s.categoria for s in startups]
    return len(categories) == len(set(categories))

def calculate_complementarity(startups: List[Startup]) -> float:
    """Calculate average pairwise semantic similarity within a cluster."""
    if len(startups) < 2:
        return 0.0
    similarities = []
    for i in range(len(startups)):
        for j in range(i + 1, len(startups)):
            sim = compute_similarity(startups[i].embedding, startups[j].embedding)
            similarities.append(sim)
    return sum(similarities) / len(similarities) if similarities else 0.0

def generate_candidate_clusters(startups: List[Startup], min_size: int, max_size: int) -> List[List[Startup]]:
    """Generate valid candidate clusters respecting category constraint."""
    candidates = []
    by_category = {}
    for startup in startups:
        if startup.categoria not in by_category:
            by_category[startup.categoria] = []
        by_category[startup.categoria].append(startup)
    
    categories = list(by_category.keys())
    if len(categories) >= min_size:
        def build_from_categories(selected_categories, current_cluster):
            if len(current_cluster) >= max_size:
                if len(current_cluster) >= min_size:
                    candidates.append(current_cluster[:])
                return
            for cat in selected_categories:
                if cat in by_category:
                    for startup in by_category[cat]:
                        if startup not in current_cluster:
                            current_cluster.append(startup)
                            remaining = [c for c in selected_categories if c != cat]
                            build_from_categories(remaining, current_cluster)
                            current_cluster.pop()
                            if len(candidates) >= 10:
                                return
        build_from_categories(categories, [])
    
    if not candidates and len(startups) >= min_size:
        embeddings = np.array([s.embedding for s in startups])
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(embeddings_scaled)
        clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='precomputed').fit(distances)
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:
                continue
            cluster_startups = [startups[i] for i in range(len(startups)) if clustering.labels_[i] == cluster_id]
            if is_valid_cluster(cluster_startups) and min_size <= len(cluster_startups) <= max_size:
                candidates.append(cluster_startups)
    
    return candidates

def select_best_clusters(candidates: List[List[Startup]]) -> List[List[Startup]]:
    """Select the best non-overlapping clusters."""
    scored_candidates = [(calculate_complementarity(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    selected = []
    used_startups = set()
    for score, cluster in scored_candidates:
        cluster_ids = {s.id for s in cluster}
        if not (cluster_ids & used_startups):
            selected.append(cluster)
            used_startups.update(cluster_ids)
    return selected

def build_complementary_clusters(startups: List[Startup], audience_groups: Dict[str, List[Startup]], 
                                min_size: int = MIN_CLUSTER_SIZE, max_size: int = MAX_CLUSTER_SIZE) -> List[CoMarketingCluster]:
    """Build complementary clusters within each audience group."""
    clusters = []
    cluster_counter = 0
    for audience, group in audience_groups.items():
        logger.info(f"\\nClustering audience: {audience} ({len(group)} startups)")
        if len(group) < min_size:
            logger.warning(f"Audience '{audience}' has only {len(group)} startups (minimum: {min_size}). Skipping.")
            continue
        candidate_clusters = generate_candidate_clusters(group, min_size, max_size)
        if not candidate_clusters:
            continue
        selected_clusters = select_best_clusters(candidate_clusters)
        for candidate in selected_clusters:
            cluster_counter += 1
            cluster_id = f"cluster_{audience.lower().replace(' ', '_')}_{cluster_counter}"
            complementarity_score = calculate_complementarity(candidate)
            categories = ", ".join([s.categoria for s in candidate])
            co_cluster = CoMarketingCluster(
                cluster_id=cluster_id,
                audiencia_objetivo=audience,
                startups=[s.to_dict() for s in candidate],
                complementarity_score=complementarity_score,
                rationale=f"Complementary tools for {audience}: {categories}"
            )
            clusters.append(co_cluster)
            logger.info(f"  Created {cluster_id} with {len(candidate)} startups (score: {complementarity_score:.4f})")
    return clusters

def save_clusters(clusters: List[CoMarketingCluster], output_path: str = "clusters_result.json") -> None:
    """Save clustering results to JSON file."""
    output_data = {
        'generated_at': str(Path(__file__).stat().st_mtime),
        'total_clusters': len(clusters),
        'clusters': [c.to_dict() for c in clusters]
    }
    Path(output_path).write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info(f"Clusters saved to {output_path}")

def print_summary(clusters: List[CoMarketingCluster]) -> None:
    """Print a summary of clustering results to console."""
    print("\\n" + "="*70)
    print("CO-MARKETING CLUSTERING RESULTS")
    print("="*70)
    print(f"Total Clusters: {len(clusters)}\\n")
    for cluster in clusters:
        print(f"Cluster ID: {cluster.cluster_id}")
        print(f"Audience: {cluster.audiencia_objetivo}")
        print(f"Complementarity Score: {cluster.complementarity_score:.4f}")
        print(f"Startups:")
        for startup in cluster.startups:
            print(f"  - {startup['nombre']} ({startup['categoria']})")
        print()

def main(data_source: Optional[str] = None, output_path: str = "clusters_result.json") -> None:
    """Execute the complete co-marketing clustering pipeline."""
    logger.info("Starting Co-Marketing Clustering Pipeline...")
    startups = load_startup_data(data_source)
    startups = generate_embeddings(startups)
    audience_groups = cluster_by_audience(startups)
    clusters = build_complementary_clusters(startups, audience_groups)
    save_clusters(clusters, output_path)
    print_summary(clusters)
    logger.info(f"Pipeline completed successfully. Generated {len(clusters)} clusters.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Market Basket Analysis for Co-Marketing Clusters")
    parser.add_argument("--data", type=str, default=None, help="Path to JSON file with startup data")
    parser.add_argument("--output", type=str, default="clusters_result.json", help="Path to output JSON file")
    args = parser.parse_args()
    try:
        main(data_source=args.data, output_path=args.output)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
