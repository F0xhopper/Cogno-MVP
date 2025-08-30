// Simple test script for advanced RAG functionality
// Run with: node test-advanced-rag.js

import { getAdvancedRAGConfig } from "./src/config/advanced-rag.config.js";

console.log("Testing Advanced RAG Configuration...\n");

try {
  const config = getAdvancedRAGConfig();
  console.log("✅ Configuration loaded successfully");
  console.log("Query Expansion enabled:", config.queryExpansion.enabled);
  console.log("Self-Critique enabled:", config.selfCritique.enabled);
  console.log("Reranking enabled:", config.reranking.enabled);
  console.log("Hybrid Search enabled:", config.hybridSearch.enabled);
  console.log("Critique threshold:", config.selfCritique.threshold);
  console.log("Max context documents:", config.synthesis.maxContextDocuments);

  console.log("\n🎯 Advanced RAG is ready to use!");
  console.log("\nTo test it:");
  console.log("1. Set your COHERE_API_KEY environment variable");
  console.log("2. Start the server: npm run dev");
  console.log("3. Use the /query/advanced endpoint for detailed analysis");
  console.log("4. Compare with the regular /query endpoint");
} catch (error) {
  console.error("❌ Error loading configuration:", error.message);
}
