import OpenAI from "openai";
import { getAdvancedRAGConfig } from "../config/advanced-rag.config.js";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_API_KEY) {
  throw new Error("Missing OPENAI_API_KEY");
}

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

const CHAT_MODEL = process.env.GPT_MODEL || "gpt-4o-mini";

export interface QueryDetail {
  query: string;
  content: string[];
  critiqueScore: number;
  critiqueDetails: Record<string, any>;
  retrievalNeeded: boolean;
  searchNeeded: boolean;
}

export interface AdvancedRAGResult {
  query: string;
  expandedQueries: string[];
  documents: any[];
  rerankedDocuments: any[];
  finalAnswer: string;
  critiqueScore: number;
  confidence: number;
}

/**
 * Generate multiple derivative queries from the original query
 * This implements the RAG-Fusion technique for broader information retrieval
 */
export async function generateExpandedQueries(
  originalQuery: string,
  numQueries: number = 3
): Promise<string[]> {
  const config = getAdvancedRAGConfig();

  if (!config.queryExpansion.enabled) {
    return [originalQuery];
  }

  const prompt = `Given the following user query, generate ${numQueries} different but related queries that would help retrieve comprehensive information. 
  Make each query focus on different aspects or perspectives of the original question.
  
  Original Query: "${originalQuery}"
  
  Generate ${numQueries} expanded queries (one per line):`;

  const completion = await openai.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      {
        role: "system",
        content:
          "You are an expert at expanding queries for comprehensive information retrieval. Generate diverse, focused queries.",
      },
      { role: "user", content: prompt },
    ],
    temperature: config.queryExpansion.temperature,
  });

  const response = completion.choices[0]?.message?.content?.trim() ?? "";
  const queries = response
    .split("\n")
    .map((q) => q.trim())
    .filter((q) => q.length > 0)
    .slice(0, numQueries);

  // Always include the original query
  return [originalQuery, ...queries];
}


/**
 * Critique the quality and relevance of retrieved documents and generated response
 * This implements the Self-RAG critique mechanism
 */
export async function critiqueResponse(
  query: string,
  response: string,
  context?: string[]
): Promise<{ score: number; details: Record<string, any> }> {
  const contextInfo = context ? `\nContext used: ${context.join(" | ")}` : "";

  const prompt = `Critically evaluate the following response to the user query. Rate it from 0.0 to 1.0 and provide detailed feedback.
  
  Query: "${query}"
  Response: "${response}"${contextInfo}
  
  Evaluate on these criteria:
  1. Relevance to the query (0.0-1.0)
  2. Factual accuracy (0.0-1.0)
  3. Completeness of answer (0.0-1.0)
  4. Clarity and coherence (0.0-1.0)
  
  Respond in this exact JSON format:
  {
    "score": 0.85,
    "details": {
      "relevance": 0.9,
      "accuracy": 0.8,
      "completeness": 0.85,
      "clarity": 0.85,
      "feedback": "Detailed feedback here"
    }
  }`;

  try {
    const completion = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        {
          role: "system",
          content: "You are an expert evaluator. Respond only with valid JSON.",
        },
        { role: "user", content: prompt },
      ],
      temperature: 0.1,
    });

    const responseText = completion.choices[0]?.message?.content?.trim() ?? "";
    const parsed = JSON.parse(responseText);

    return {
      score: parsed.score || 0.0,
      details: parsed.details || {},
    };
  } catch (error) {
    console.error("Error parsing critique response:", error);
    return {
      score: 0.5,
      details: { error: "Failed to parse critique" },
    };
  }
}

/**
 * Rerank documents using Cohere's reranker for better relevance
 * This implements the reranking technique mentioned in the article
 */
export async function rerankDocuments(
  query: string,
  documents: string[],
  topN: number = 5
): Promise<{ text: string; relevanceScore: number }[]> {
  const config = getAdvancedRAGConfig();

  if (!config.reranking.enabled) {
    // Return documents with simple ranking if reranking is disabled
    return documents.map((doc, index) => ({
      text: doc,
      relevanceScore: 1.0 - index * 0.1,
    }));
  }

  // Use Pinecone's built-in reranking (already done in the search)
  // Just return documents with relevance scores based on their order
  // The actual reranking happens in the Pinecone search with the rerank parameter
  return documents.map((doc, index) => ({
    text: doc,
    relevanceScore: 1.0 - index * 0.1, // Higher score for earlier documents
  }));
}

/**
 * Synthesize a comprehensive answer from multiple sources
 * This implements better context integration and synthesis
 */
export async function synthesizeAnswer(
  query: string,
  documents: string[],
  critiqueScore: number
): Promise<string> {
  const config = getAdvancedRAGConfig();
  const contextLimit = Math.min(
    documents.length,
    config.synthesis.maxContextDocuments
  );
  const selectedDocs = documents.slice(0, contextLimit);

  let prompt = `You are an expert on St Thomas Aquinas and philosophical topics. 
  Using the context below, provide a comprehensive and accurate answer to the user's question.
  
  Context (${selectedDocs.length} documents):
  ${selectedDocs.map((doc, i) => `[${i + 1}] ${doc}`).join("\n\n")}
  
  Question: ${query}
  
  Instructions:
  - Base your answer primarily on the provided context
  - If the context doesn't contain enough information, acknowledge this limitation
  - Be precise and cite specific parts of the context when relevant
  - Maintain academic rigor while being accessible`;

  if (config.synthesis.includeCitations) {
    prompt += `\n- Include citations to specific document numbers when referencing information`;
  }

  prompt += `\n\nAnswer:`;

  const completion = await openai.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      {
        role: "system",
        content:
          "You are a scholarly expert on St Thomas Aquinas and philosophy. Provide accurate, well-reasoned answers based on the given context.",
      },
      { role: "user", content: prompt },
    ],
    temperature: config.synthesis.temperature,
    max_tokens: 4000, // Increased to ensure full answers
  });

  const answer = completion.choices[0]?.message?.content?.trim() ?? "";
  console.log("Generated answer length:", answer.length);

  return answer;
}

/**
 * Main advanced RAG function that orchestrates all techniques
 */
export async function advancedRAGQuery(
  query: string,
  vectorSearchResults: any[],
  topK: number = 5
): Promise<AdvancedRAGResult> {
  console.log("Starting advanced RAG processing for query:", query);

  const config = getAdvancedRAGConfig();

  // Step 1: Query Expansion
  const expandedQueries = await generateExpandedQueries(
    query,
    config.queryExpansion.numExpandedQueries
  );
  console.log("Generated expanded queries:", expandedQueries);


  

  // Step 3: Extract document texts from vector search results
  console.log("Raw vector search results:", vectorSearchResults.length);

  const documents = vectorSearchResults
    .map((result: any) => (result.fields as any)?.chunk_text || "")
    .filter(Boolean);

  console.log("Extracted documents:", documents.length);
  if (documents.length > 0) {
    console.log("First document preview:", documents[0].substring(0, 200));
  }

  if (documents.length === 0) {
    const answer =
      "I don't have enough information in my knowledge base to answer this question accurately.";
    return {
      query,
      expandedQueries: [query],
      documents: [],
      rerankedDocuments: [],
      finalAnswer: answer,
      critiqueScore: 0.0,
      confidence: 0.0,
    };
  }

  // Step 4: Rerank documents for better relevance
  console.log("About to rerank", documents.length, "documents");
  const rerankedDocuments = await rerankDocuments(query, documents, topK);
  console.log("Reranked documents:", rerankedDocuments.length);
  console.log("Sample reranked document:", rerankedDocuments[0]);

  // Step 5: Generate initial answer
  console.log(
    "Generating answer with",
    rerankedDocuments.length,
    "reranked documents"
  );
  const initialAnswer = await synthesizeAnswer(
    query,
    rerankedDocuments.map((d) => d.text),
    1.0
  );

  console.log("Initial answer length:", initialAnswer.length);
  console.log("Initial answer preview:", initialAnswer.substring(0, 200));

  // Step 6: Critique the response
  const critique = await critiqueResponse(
    query,
    initialAnswer,
    rerankedDocuments.map((d) => d.text)
  );
  console.log("Critique score:", critique.score);

  // Step 7: If critique score is low, try to improve the answer
  let finalAnswer = initialAnswer;
  let confidence = critique.score;

  if (critique.score < config.selfCritique.threshold) {
    console.log(
      `Low critique score (${critique.score}), attempting to improve answer...`
    );

    let improvementAttempts = 0;
    let bestAnswer = initialAnswer;
    let bestScore = critique.score;

    while (improvementAttempts < config.selfCritique.maxImprovementAttempts) {
      improvementAttempts++;

      // Try to generate a better answer with more focused context
      const topDocuments = rerankedDocuments
        .filter((d) => d.relevanceScore > 0.7)
        .map((d) => d.text);

      if (topDocuments.length > 0) {
        const improvedAnswer = await synthesizeAnswer(
          query,
          topDocuments,
          bestScore
        );
        const improvedCritique = await critiqueResponse(
          query,
          improvedAnswer,
          topDocuments
        );

        if (improvedCritique.score > bestScore) {
          bestAnswer = improvedAnswer;
          confidence = improvedCritique.score;
          console.log(
            `Answer improved on attempt ${improvementAttempts}, new score: ${improvedCritique.score}`
          );
        }
      }

      // Stop if we've reached a good score
      if (bestScore >= config.selfCritique.threshold) {
        break;
      }
    }

    finalAnswer = bestAnswer;
    confidence = bestScore;
  }

  return {
    query,
    expandedQueries,
    documents,
    rerankedDocuments,
    finalAnswer,
    critiqueScore: critique.score,
    confidence,
  };
}
