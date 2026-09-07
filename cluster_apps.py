#!/usr/bin/env python3
"""
Market Basket Analysis & Semantic Clustering para CoTraffic.

Arma "clusters de co-marketing" de 4 a 5 startups complementarias bajo dos
restricciones duras:

  1. Todas las startups de un cluster comparten la misma audiencia objetivo.
  2. Ninguna categoria se repite dentro del cluster, para que el grupo no
     incluya competidores directos.

Entre los grupos que cumplen ambas reglas se elige el de mayor cohesion
semantica, medida como la similitud coseno promedio entre los embeddings de
las descripciones (Sentence-Transformers + scikit-learn).

Fuentes de datos (ver --data / --rss), en orden de precedencia:
  - JSON local o remoto (HTTPS) con la lista de startups.
  - Feeds RSS/Atom publicos (Product Hunt, BetaList, etc.), infiriendo
    categoria y audiencia por palabras clave.
  - Dataset de ejemplo embebido, usado como fallback.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Parametros de negocio
# --------------------------------------------------------------------------

MIN_CLUSTER_SIZE = 4
MAX_CLUSTER_SIZE = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# La busqueda exhaustiva de combinaciones es optima pero crece como C(n, k).
# Por encima de este numero de combinaciones posibles pasamos a construccion
# greedy, que es lineal en la cantidad de semillas y escala a datasets grandes.
EXHAUSTIVE_LIMIT = 50_000

# Taxonomia usada para inferir categoria y audiencia cuando la fuente de datos
# no las trae (tipicamente RSS). Son heuristicas por palabra clave: alcanzan
# para prototipar, no reemplazan una clasificacion curada.
CATEGORY_KEYWORDS: Dict[str, Sequence[str]] = {
    "Productividad": ("task", "todo", "productiv", "workflow", "note", "focus"),
    "Tracking": ("time track", "timesheet", "tracking", "pomodoro"),
    "Ventas": ("sales", "proposal", "pipeline", "lead", "outreach"),
    "CRM": ("crm", "client", "customer relation", "contact"),
    "Facturacion": ("invoice", "billing", "payment", "factur"),
    "Marketing": ("marketing", "campaign", "seo", "newsletter", "ads"),
    "E-commerce": ("ecommerce", "e-commerce", "shopify", "storefront", "checkout"),
    "Inventario": ("inventory", "stock", "warehouse"),
    "Logistica": ("shipping", "logistic", "delivery", "fulfillment"),
    "DevOps": ("devops", "ci/cd", "deploy", "kubernetes"),
    "QA": ("bug", "testing", "test suite", "flaky"),
    "Documentacion": ("documentation", "docs", "knowledge base", "wiki"),
    "Observability": ("monitoring", "observability", "logging", "apm", "uptime"),
    "Seguridad": ("security", "secret", "vault", "encryption"),
    "Analytics": ("analytics", "dashboard", "metrics", "insight"),
}

AUDIENCE_KEYWORDS: Dict[str, Sequence[str]] = {
    "Freelancers & Remote Teams": ("freelance", "solopreneur", "remote team", "contractor"),
    "E-commerce Stores": ("ecommerce", "e-commerce", "online store", "shopify", "merchant"),
    "Developers & Dev Teams": ("developer", "engineer", "devops", "api", "sdk", "code"),
    "Agencias": ("agency", "agencies", "studio", "client work"),
}

UNKNOWN_CATEGORY = "Sin categoria"
UNKNOWN_AUDIENCE = "Sin audiencia"


# --------------------------------------------------------------------------
# Modelo de datos
# --------------------------------------------------------------------------

# eq=False mantiene la igualdad por identidad: evita que dataclass compare los
# embeddings (arrays de numpy) al evaluar `in` sobre listas de startups.
@dataclass(eq=False)
class Startup:
    """Una startup/app con los metadatos relevantes para el agrupamiento."""

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
        data.pop("embedding", None)
        return data

    def texto_semantico(self) -> str:
        """Texto que se convierte en embedding."""
        tags = ", ".join(self.tags)
        return f"{self.descripcion}. Tags: {tags}" if tags else self.descripcion


@dataclass
class CoMarketingCluster:
    """Grupo de startups complementarias listo para una campana conjunta."""

    cluster_id: str
    audiencia_objetivo: str
    startups: List[Dict]
    complementarity_score: float
    rationale: str

    def to_dict(self) -> Dict:
        return {
            "cluster_id": self.cluster_id,
            "audiencia_objetivo": self.audiencia_objetivo,
            "startups": self.startups,
            "complementarity_score": round(self.complementarity_score, 4),
            "rationale": self.rationale,
            "size": len(self.startups),
        }


# --------------------------------------------------------------------------
# Ingesta de datos
# --------------------------------------------------------------------------


def load_startup_data(
    data_source: Optional[str] = None,
    rss_feeds: Optional[Sequence[str]] = None,
) -> List[Startup]:
    """Carga startups desde JSON (local o remoto), RSS o el dataset de ejemplo."""
    if data_source:
        startups = _load_from_json_source(data_source)
        if startups:
            logger.info(f"{len(startups)} startups cargadas desde {data_source}")
            return startups
        logger.warning("La fuente JSON no devolvio datos; se intenta la siguiente opcion")

    if rss_feeds:
        startups = _load_from_rss(rss_feeds)
        if startups:
            return startups
        logger.warning("Los feeds RSS no devolvieron datos; se usa el dataset de ejemplo")

    logger.info("Usando el dataset de ejemplo embebido")
    return _example_startups()


def _load_from_json_source(source: str) -> List[Startup]:
    """Lee una lista de startups desde un path local o una URL HTTP(S)."""
    try:
        if source.startswith(("http://", "https://")):
            logger.info(f"Descargando dataset desde {source}")
            with urllib.request.urlopen(source, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        else:
            path = Path(source)
            if not path.exists():
                logger.warning(f"No existe el archivo {source}")
                return []
            logger.info(f"Leyendo dataset local {source}")
            payload = json.loads(path.read_text(encoding="utf-8"))

        # Se acepta tanto una lista suelta como {"startups": [...]}.
        items = payload.get("startups", []) if isinstance(payload, dict) else payload
        return [_startup_from_dict(item) for item in items]
    except Exception as exc:
        logger.error(f"No se pudo leer {source}: {exc}")
        return []


def _startup_from_dict(item: Dict) -> Startup:
    """Construye una Startup tolerando campos faltantes en la fuente."""
    nombre = item.get("nombre") or item.get("name") or "Sin nombre"
    return Startup(
        id=str(item.get("id") or nombre),
        nombre=nombre,
        categoria=item.get("categoria") or UNKNOWN_CATEGORY,
        descripcion=item.get("descripcion") or item.get("description") or "",
        tags=list(item.get("tags") or []),
        audiencia_objetivo=item.get("audiencia_objetivo") or UNKNOWN_AUDIENCE,
        url=item.get("url"),
    )


def _load_from_rss(feed_urls: Sequence[str]) -> List[Startup]:
    """Extrae startups de feeds RSS/Atom publicos (Product Hunt, BetaList, ...)."""
    try:
        import feedparser
    except ImportError:
        logger.error("feedparser no esta instalado: no se pueden leer feeds RSS")
        return []

    startups: List[Startup] = []
    for feed_url in feed_urls:
        logger.info(f"Leyendo feed {feed_url}")
        parsed = feedparser.parse(feed_url)
        if not parsed.entries:
            logger.warning(f"Feed sin entradas o ilegible: {feed_url}")
            continue

        for entry in parsed.entries:
            nombre = (entry.get("title") or "").strip()
            if not nombre:
                continue
            descripcion = _strip_html(entry.get("summary") or entry.get("description") or "")
            tags = [t.get("term", "") for t in entry.get("tags", []) if t.get("term")]
            texto = f"{nombre} {descripcion} {' '.join(tags)}"
            startups.append(
                Startup(
                    id=str(entry.get("id") or entry.get("link") or nombre),
                    nombre=nombre,
                    categoria=_infer_label(texto, CATEGORY_KEYWORDS, UNKNOWN_CATEGORY),
                    descripcion=descripcion,
                    tags=tags,
                    audiencia_objetivo=_infer_label(texto, AUDIENCE_KEYWORDS, UNKNOWN_AUDIENCE),
                    url=entry.get("link"),
                )
            )

    logger.info(f"{len(startups)} entradas obtenidas de {len(feed_urls)} feed(s)")
    return startups


def _strip_html(texto: str) -> str:
    """Limpia el HTML que suelen traer los resumenes de RSS."""
    return re.sub(r"<[^>]+>", " ", texto).replace("&nbsp;", " ").strip()


def _infer_label(texto: str, taxonomia: Dict[str, Sequence[str]], default: str) -> str:
    """Devuelve la etiqueta cuyas palabras clave aparecen mas veces en el texto."""
    texto = texto.lower()
    mejor, mejor_hits = default, 0
    for etiqueta, palabras in taxonomia.items():
        hits = sum(texto.count(palabra) for palabra in palabras)
        if hits > mejor_hits:
            mejor, mejor_hits = etiqueta, hits
    return mejor


def _example_startups() -> List[Startup]:
    """Dataset simulado: 3 audiencias x 7 startups, con categorias repetidas a
    proposito para ejercitar la restriccion de no-competencia."""
    crudo = [
        # --- Freelancers & Remote Teams ---
        ("1", "TaskFlow", "Productividad", "Gestion de tareas colaborativa para equipos remotos",
         ["tareas", "colaboracion", "equipos"], "Freelancers & Remote Teams"),
        ("2", "TimeLogger", "Tracking", "Time tracking automatizado para freelancers",
         ["time-tracking", "facturacion", "freelance"], "Freelancers & Remote Teams"),
        ("3", "ProposalMaker", "Ventas", "Generador de propuestas profesionales con plantillas",
         ["propuestas", "B2B", "ventas"], "Freelancers & Remote Teams"),
        ("4", "ClientPortal", "CRM", "Portal seguro para la comunicacion cliente-proveedor",
         ["comunicacion", "cliente", "gestion"], "Freelancers & Remote Teams"),
        ("5", "InvoiceZen", "Facturacion", "Facturacion recurrente y cobros para trabajo independiente",
         ["facturas", "cobros", "freelance"], "Freelancers & Remote Teams"),
        ("6", "FocusRoom", "Productividad", "Sesiones de trabajo profundo con bloqueo de distracciones",
         ["foco", "pomodoro", "habitos"], "Freelancers & Remote Teams"),
        ("7", "ContractSign", "Legal", "Contratos y firma electronica para acuerdos con clientes",
         ["contratos", "firma", "legal"], "Freelancers & Remote Teams"),
        # --- E-commerce Stores ---
        ("8", "ShopAI", "E-commerce", "Plataforma de e-commerce con IA para recomendar productos",
         ["e-commerce", "IA", "recomendaciones"], "E-commerce Stores"),
        ("9", "InvTracker", "Inventario", "Gestion de inventario en tiempo real multi-deposito",
         ["inventario", "stock", "e-commerce"], "E-commerce Stores"),
        ("10", "ReviewBooster", "Marketing", "Recopila y muestra resenas de clientes verificadas",
         ["reviews", "social-proof", "marketing"], "E-commerce Stores"),
        ("11", "ShippingPro", "Logistica", "Optimizacion de envios y costos multicarrier",
         ["envios", "logistica", "e-commerce"], "E-commerce Stores"),
        ("12", "CartRescue", "Retencion", "Recupera carritos abandonados con secuencias automaticas",
         ["carrito", "retencion", "email"], "E-commerce Stores"),
        ("13", "PricePulse", "Pricing", "Monitoreo de precios de la competencia y repricing",
         ["precios", "competencia", "margen"], "E-commerce Stores"),
        ("14", "AdSpark", "Marketing", "Creacion y testeo de anuncios para tiendas online",
         ["ads", "creatividades", "performance"], "E-commerce Stores"),
        # --- Developers & Dev Teams ---
        ("15", "CodeDeploy", "DevOps", "Pipeline CI/CD simplificado para equipos de desarrollo",
         ["CI/CD", "deployment", "DevOps"], "Developers & Dev Teams"),
        ("16", "BugTracker", "QA", "Seguimiento de errores integrado al flujo de trabajo",
         ["bugs", "testing", "QA"], "Developers & Dev Teams"),
        ("17", "DocHub", "Documentacion", "Plataforma colaborativa para documentacion de APIs",
         ["documentacion", "API", "colaboracion"], "Developers & Dev Teams"),
        ("18", "MonitorPro", "Observability", "Monitoreo de aplicaciones y analisis de performance",
         ["monitoreo", "performance", "observability"], "Developers & Dev Teams"),
        ("19", "SecretVault", "Seguridad", "Gestion de secretos y credenciales para entornos cloud",
         ["secretos", "seguridad", "cloud"], "Developers & Dev Teams"),
        ("20", "APIForge", "API Design", "Diseno y mocking de APIs antes de escribir codigo",
         ["API", "diseno", "mocking"], "Developers & Dev Teams"),
        ("21", "FlakyGuard", "QA", "Detecta y aisla tests inestables en la suite automatizada",
         ["tests", "flaky", "CI"], "Developers & Dev Teams"),
    ]

    startups = [
        Startup(
            id=sid,
            nombre=nombre,
            categoria=categoria,
            descripcion=descripcion,
            tags=tags,
            audiencia_objetivo=audiencia,
            url=f"https://{nombre.lower()}.example.com",
        )
        for sid, nombre, categoria, descripcion, tags, audiencia in crudo
    ]
    logger.info(f"{len(startups)} startups de ejemplo cargadas")
    return startups


# --------------------------------------------------------------------------
# Embeddings y similitud
# --------------------------------------------------------------------------


def generate_embeddings(startups: List[Startup], model_name: str = EMBEDDING_MODEL) -> List[Startup]:
    """Calcula el embedding semantico de cada startup a partir de su descripcion."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Cargando modelo de embeddings: {model_name}")
    model = SentenceTransformer(model_name)

    textos = [s.texto_semantico() for s in startups]
    logger.info(f"Generando embeddings para {len(startups)} startups...")
    embeddings = model.encode(textos, show_progress_bar=False)

    for startup, embedding in zip(startups, embeddings):
        startup.embedding = embedding
    logger.info("Embeddings generados correctamente")
    return startups


def build_similarity_index(startups: List[Startup]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Matriz de similitud coseno entre todas las startups + indice id -> fila."""
    matriz = np.vstack([s.embedding for s in startups])
    similitudes = cosine_similarity(matriz)
    indice = {s.id: i for i, s in enumerate(startups)}
    return similitudes, indice


def calculate_complementarity(
    cluster: Sequence[Startup], similitudes: np.ndarray, indice: Dict[str, int]
) -> float:
    """Cohesion del cluster: similitud coseno promedio entre todos sus pares.

    Como el cluster ya tiene garantizada una audiencia comun y categorias
    distintas, una similitud alta indica startups del mismo universo tematico
    pero con funciones diferentes: exactamente el perfil complementario buscado.
    """
    pares = list(itertools.combinations(cluster, 2))
    if not pares:
        return 0.0
    valores = [similitudes[indice[a.id], indice[b.id]] for a, b in pares]
    return float(np.mean(valores))


# --------------------------------------------------------------------------
# Construccion de clusters
# --------------------------------------------------------------------------


def cluster_by_audience(startups: List[Startup]) -> Dict[str, List[Startup]]:
    """Agrupa startups por audiencia objetivo (restriccion 1)."""
    grupos: Dict[str, List[Startup]] = {}
    for startup in startups:
        grupos.setdefault(startup.audiencia_objetivo, []).append(startup)
    logger.info(f"Se encontraron {len(grupos)} audiencias distintas")
    return grupos


def has_unique_categories(cluster: Sequence[Startup]) -> bool:
    """Valida la restriccion 2: ninguna categoria repetida en el cluster."""
    categorias = [s.categoria for s in cluster]
    return len(categorias) == len(set(categorias))


def generate_candidate_clusters(
    grupo: List[Startup],
    similitudes: np.ndarray,
    indice: Dict[str, int],
    min_size: int = MIN_CLUSTER_SIZE,
    max_size: int = MAX_CLUSTER_SIZE,
) -> List[List[Startup]]:
    """Genera grupos candidatos de tamano valido y sin categorias repetidas.

    Para grupos chicos hace busqueda exhaustiva (optima); para grupos grandes
    cae en construccion greedy para no explotar combinatoriamente.
    """
    if len(grupo) < min_size:
        return []

    tope = min(max_size, len(grupo))
    combinaciones = sum(math.comb(len(grupo), k) for k in range(min_size, tope + 1))

    if combinaciones <= EXHAUSTIVE_LIMIT:
        candidatos = _exhaustive_candidates(grupo, min_size, tope)
        logger.info(f"  Busqueda exhaustiva: {len(candidatos)} candidatos validos")
    else:
        candidatos = _greedy_candidates(grupo, similitudes, indice, min_size, tope)
        logger.info(f"  Busqueda greedy: {len(candidatos)} candidatos validos")
    return candidatos


def _exhaustive_candidates(
    grupo: List[Startup], min_size: int, max_size: int
) -> List[List[Startup]]:
    """Todas las combinaciones de tamano [min_size, max_size] sin categorias repetidas."""
    candidatos = []
    for size in range(max_size, min_size - 1, -1):
        for combo in itertools.combinations(grupo, size):
            if has_unique_categories(combo):
                candidatos.append(list(combo))
    return candidatos


def _greedy_candidates(
    grupo: List[Startup],
    similitudes: np.ndarray,
    indice: Dict[str, int],
    min_size: int,
    max_size: int,
) -> List[List[Startup]]:
    """Desde cada startup semilla, suma la vecina mas similar de categoria nueva."""
    candidatos: List[List[Startup]] = []
    vistos = set()

    for semilla in grupo:
        cluster = [semilla]
        categorias = {semilla.categoria}

        while len(cluster) < max_size:
            mejor, mejor_score = None, -1.0
            for candidata in grupo:
                if candidata in cluster or candidata.categoria in categorias:
                    continue
                score = float(
                    np.mean([similitudes[indice[candidata.id], indice[m.id]] for m in cluster])
                )
                if score > mejor_score:
                    mejor, mejor_score = candidata, score
            if mejor is None:
                break
            cluster.append(mejor)
            categorias.add(mejor.categoria)

        if len(cluster) >= min_size:
            firma = frozenset(s.id for s in cluster)
            if firma not in vistos:
                vistos.add(firma)
                candidatos.append(cluster)

    return candidatos


def select_best_clusters(
    candidatos: List[List[Startup]], similitudes: np.ndarray, indice: Dict[str, int]
) -> List[Tuple[List[Startup], float]]:
    """Elige los mejores clusters sin solapar startups entre si.

    Ordena por tamano (un cluster de 5 vale mas que uno de 4 para una campana
    conjunta) y, a igual tamano, por cohesion semantica.
    """
    puntuados = [
        (len(c), calculate_complementarity(c, similitudes, indice), c) for c in candidatos
    ]
    puntuados.sort(key=lambda t: (t[0], t[1]), reverse=True)

    seleccionados: List[Tuple[List[Startup], float]] = []
    usadas: set = set()
    for _, score, cluster in puntuados:
        ids = {s.id for s in cluster}
        if ids & usadas:
            continue
        seleccionados.append((cluster, score))
        usadas |= ids
    return seleccionados


def build_complementary_clusters(
    startups: List[Startup],
    similitudes: np.ndarray,
    indice: Dict[str, int],
    min_size: int = MIN_CLUSTER_SIZE,
    max_size: int = MAX_CLUSTER_SIZE,
) -> List[CoMarketingCluster]:
    """Pipeline de agrupamiento: audiencia -> candidatos -> seleccion final."""
    clusters: List[CoMarketingCluster] = []
    contador = 0

    for audiencia, grupo in cluster_by_audience(startups).items():
        logger.info(f"Procesando audiencia '{audiencia}' ({len(grupo)} startups)")

        if len(grupo) < min_size:
            logger.warning(
                f"  '{audiencia}' tiene solo {len(grupo)} startups (minimo {min_size}). Se omite."
            )
            continue

        candidatos = generate_candidate_clusters(grupo, similitudes, indice, min_size, max_size)
        if not candidatos:
            logger.warning(f"  '{audiencia}' no permite ningun cluster sin categorias repetidas")
            continue

        for cluster, score in select_best_clusters(candidatos, similitudes, indice):
            contador += 1
            slug = re.sub(r"[^a-z0-9]+", "_", audiencia.lower()).strip("_")
            categorias = ", ".join(s.categoria for s in cluster)
            clusters.append(
                CoMarketingCluster(
                    cluster_id=f"cluster_{slug}_{contador}",
                    audiencia_objetivo=audiencia,
                    startups=[s.to_dict() for s in cluster],
                    complementarity_score=score,
                    rationale=f"Herramientas complementarias para {audiencia}: {categorias}",
                )
            )
            logger.info(
                f"  {clusters[-1].cluster_id}: {len(cluster)} startups (cohesion {score:.4f})"
            )

    return clusters


# --------------------------------------------------------------------------
# Salida
# --------------------------------------------------------------------------


def save_clusters(clusters: List[CoMarketingCluster], output_path: str) -> None:
    """Persiste el resultado en JSON."""
    salida = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_clusters": len(clusters),
        "clusters": [c.to_dict() for c in clusters],
    }
    Path(output_path).write_text(
        json.dumps(salida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(f"Resultado guardado en {output_path}")


def print_summary(clusters: List[CoMarketingCluster]) -> None:
    """Resumen legible por consola, util en los logs de GitHub Actions."""
    print("\n" + "=" * 70)
    print("CLUSTERS DE CO-MARKETING")
    print("=" * 70)
    print(f"Total de clusters: {len(clusters)}\n")
    for cluster in clusters:
        print(f"{cluster.cluster_id}  ({len(cluster.startups)} startups)")
        print(f"  Audiencia : {cluster.audiencia_objetivo}")
        print(f"  Cohesion  : {cluster.complementarity_score:.4f}")
        for startup in cluster.startups:
            print(f"    - {startup['nombre']} ({startup['categoria']})")
        print()


def main(
    data_source: Optional[str] = None,
    rss_feeds: Optional[Sequence[str]] = None,
    output_path: str = "clusters_result.json",
) -> List[CoMarketingCluster]:
    """Ejecuta el pipeline completo de clustering de co-marketing."""
    logger.info("Iniciando pipeline de clustering de co-marketing...")

    startups = load_startup_data(data_source, rss_feeds)
    startups = generate_embeddings(startups)
    similitudes, indice = build_similarity_index(startups)
    clusters = build_complementary_clusters(startups, similitudes, indice)

    save_clusters(clusters, output_path)
    print_summary(clusters)
    logger.info(f"Pipeline completado. Se generaron {len(clusters)} clusters.")
    return clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Market Basket Analysis para clusters de co-marketing"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path local o URL HTTPS a un JSON con la lista de startups",
    )
    parser.add_argument(
        "--rss",
        type=str,
        nargs="*",
        default=None,
        help="Uno o mas feeds RSS/Atom publicos (Product Hunt, BetaList, ...)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="clusters_result.json",
        help="Path del JSON de salida (por defecto, la raiz del repositorio)",
    )
    args = parser.parse_args()

    try:
        main(data_source=args.data, rss_feeds=args.rss, output_path=args.output)
    except Exception as exc:
        logger.error(f"El pipeline fallo: {exc}", exc_info=True)
        sys.exit(1)
