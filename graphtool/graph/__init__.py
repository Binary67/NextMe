"""Knowledge graph generation and storage."""

from graphtool.graph.base import KnowledgeGraphStore
from graphtool.graph.embedding_store import (
    NodeEmbeddingRecord,
    SqliteEmbeddingStore,
    SqliteGraphEmbeddingStore,
)
from graphtool.graph.extraction_store import JsonChunkExtractionStore
from graphtool.graph.combiner import combine_knowledge_graphs
from graphtool.graph.generator import generate_knowledge_graph
from graphtool.graph.document_store import SqliteGraphStore
from graphtool.graph.knowledge_base_store import SqliteKnowledgeBaseStore
from graphtool.graph.provenance import (
    filter_knowledge_graph_by_source,
    filter_knowledge_graph_by_sources,
)
from graphtool.graph.resolver import SemanticEntityResolver
from graphtool.graph.taxonomy import (
    aggregate_suggestions,
    evolve_taxonomy,
    migrate_promoted_types,
    promote_suggestions,
)
from graphtool.graph.taxonomy_stores import (
    JsonNodeTypeRegistryStore,
    JsonTaxonomyPromotionAuditStore,
    SqliteTaxonomySuggestionStore,
)
from graphtool.graph.taxonomy_types import (
    CANONICAL_NODE_TYPES,
    NodeTypeRegistry,
    TaxonomyEvolutionResult,
    TaxonomyPromotionRecord,
    TaxonomySuggestionAggregate,
    TaxonomySuggestionRecord,
    UNCLASSIFIED_NODE_TYPE,
    canonical_node_type_text,
    default_node_type_registry,
    normalize_type_name,
)
from graphtool.graph.types import (
    Edge,
    EdgeProvenance,
    GraphMetadata,
    KnowledgeGraph,
    Node,
    NodeProvenance,
)

__all__ = [
    "CANONICAL_NODE_TYPES",
    "Edge",
    "EdgeProvenance",
    "GraphMetadata",
    "JsonChunkExtractionStore",
    "SqliteEmbeddingStore",
    "SqliteGraphEmbeddingStore",
    "SqliteGraphStore",
    "SqliteKnowledgeBaseStore",
    "JsonNodeTypeRegistryStore",
    "JsonTaxonomyPromotionAuditStore",
    "SqliteTaxonomySuggestionStore",
    "KnowledgeGraph",
    "KnowledgeGraphStore",
    "NodeEmbeddingRecord",
    "NodeTypeRegistry",
    "Node",
    "NodeProvenance",
    "SemanticEntityResolver",
    "TaxonomyEvolutionResult",
    "TaxonomyPromotionRecord",
    "TaxonomySuggestionAggregate",
    "TaxonomySuggestionRecord",
    "UNCLASSIFIED_NODE_TYPE",
    "aggregate_suggestions",
    "canonical_node_type_text",
    "combine_knowledge_graphs",
    "default_node_type_registry",
    "evolve_taxonomy",
    "filter_knowledge_graph_by_source",
    "filter_knowledge_graph_by_sources",
    "generate_knowledge_graph",
    "migrate_promoted_types",
    "normalize_type_name",
    "promote_suggestions",
]
