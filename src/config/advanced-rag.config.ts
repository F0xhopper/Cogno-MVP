export interface AdvancedRAGConfig {
  // Query expansion settings
  queryExpansion: {
    enabled: boolean;
    numExpandedQueries: number;
    temperature: number;
  };

  // Self-critique settings
  selfCritique: {
    enabled: boolean;
    threshold: number; // Score below which to attempt improvement
    maxImprovementAttempts: number;
  };

  // Reranking settings (using Pinecone's built-in reranking)
  reranking: {
    enabled: boolean;
    topN: number;
    usePineconeReranking: boolean;
  };

  // Hybrid search settings
  hybridSearch: {
    enabled: boolean;
    vectorWeight: number;
    keywordWeight: number;
    reciprocalRankFusion: boolean;
  };

  // Response synthesis settings
  synthesis: {
    maxContextDocuments: number;
    temperature: number;
    includeCitations: boolean;
  };

  // Performance settings
  performance: {
    timeoutMs: number;
    maxRetries: number;
    cacheEnabled: boolean;
  };
}

export const defaultAdvancedRAGConfig: AdvancedRAGConfig = {
  queryExpansion: {
    enabled: true,
    numExpandedQueries: 3,
    temperature: 0.3,
  },

  selfCritique: {
    enabled: true,
    threshold: 0.7,
    maxImprovementAttempts: 2,
  },

  reranking: {
    enabled: true,
    topN: 10,
    usePineconeReranking: true,
  },

  hybridSearch: {
    enabled: true,
    vectorWeight: 0.7,
    keywordWeight: 0.3,
    reciprocalRankFusion: true,
  },

  synthesis: {
    maxContextDocuments: 8,
    temperature: 0.2,
    includeCitations: true,
  },

  performance: {
    timeoutMs: 30000,
    maxRetries: 3,
    cacheEnabled: false,
  },
};

/**
 * Get configuration with environment variable overrides
 */
export function getAdvancedRAGConfig(): AdvancedRAGConfig {
  const config = { ...defaultAdvancedRAGConfig };

  // Override with environment variables if present
  if (process.env.ADVANCED_RAG_QUERY_EXPANSION_ENABLED !== undefined) {
    config.queryExpansion.enabled =
      process.env.ADVANCED_RAG_QUERY_EXPANSION_ENABLED === "true";
  }

  if (process.env.ADVANCED_RAG_SELF_CRITIQUE_ENABLED !== undefined) {
    config.selfCritique.enabled =
      process.env.ADVANCED_RAG_SELF_CRITIQUE_ENABLED === "true";
  }

  if (process.env.ADVANCED_RAG_RERANKING_ENABLED !== undefined) {
    config.reranking.enabled =
      process.env.ADVANCED_RAG_RERANKING_ENABLED === "true";
  }

  if (process.env.ADVANCED_RAG_HYBRID_SEARCH_ENABLED !== undefined) {
    config.hybridSearch.enabled =
      process.env.ADVANCED_RAG_HYBRID_SEARCH_ENABLED === "true";
  }

  if (process.env.ADVANCED_RAG_CRITIQUE_THRESHOLD !== undefined) {
    const threshold = parseFloat(process.env.ADVANCED_RAG_CRITIQUE_THRESHOLD);
    if (!isNaN(threshold)) {
      config.selfCritique.threshold = threshold;
    }
  }

  if (process.env.ADVANCED_RAG_MAX_CONTEXT_DOCS !== undefined) {
    const maxDocs = parseInt(process.env.ADVANCED_RAG_MAX_CONTEXT_DOCS);
    if (!isNaN(maxDocs)) {
      config.synthesis.maxContextDocuments = maxDocs;
    }
  }

  return config;
}
