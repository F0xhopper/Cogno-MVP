import "dotenv/config";
import express, { Request, Response, NextFunction } from "express";
import multer from "multer";
import { z } from "zod";
import {
  ensurePineconeIndex,
  getPineconeIndex,
  upsertVectors,
} from "./services/pinecone.ts";
import { chunkText } from "./utils/chunk.ts";
import pdf from "pdf-parse";

function parseMetadata(metadata: Record<string, any> | undefined) {
  if (!metadata) {
    return {
      source: "unknown",
      chunkIndex: 0,
    };
  }

  return {
    ...metadata,
    source: metadata.source || "unknown",
    chunkIndex: metadata.chunkIndex || 0,
  };
}

const app = express();
app.use(express.json({ limit: "10mb" }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 },
});

const UploadSchema = z.object({
  namespace: z.string().min(1).max(128).optional(),
  chunkSize: z.number().int().min(200).max(4000).optional(),
  chunkOverlap: z.number().int().min(0).max(1000).optional(),
});

const QuerySchema = z.object({
  query: z.string().min(1),
  topK: z.number().int().min(1).max(50).optional(),
  namespace: z.string().min(1).max(128).optional(),
  withAnswer: z.boolean().optional(),
});

app.get("/health", (_req: Request, res: Response) => {
  res.json({ ok: true });
});

app.post(
  "/upload",
  upload.array("files", 10),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const body = req.body || {};

      const parsed = UploadSchema.safeParse(body);
      if (!parsed.success) {
        return res.status(400).json({
          error: "Invalid parameters",
          details: parsed.error.flatten(),
        });
      }

      const namespace = parsed.data.namespace ?? "default";
      const chunkSize = parsed.data.chunkSize ?? 1200;
      const chunkOverlap = parsed.data.chunkOverlap ?? 200;

      const files = req.files as Express.Multer.File[] | undefined;
      if (!files || files.length === 0) {
        return res.status(400).json({ error: "No files uploaded" });
      }

      const index = await ensurePineconeIndex();

      const allChunks: Array<{
        text: string;
        metadata: Record<string, unknown>;
      }> = [];

      for (const file of files) {
        if (file.mimetype !== "application/pdf") {
          return res.status(400).json({
            error: `Unsupported mimetype: ${file.mimetype}. Only PDF allowed.`,
          });
        }

        const pdfData = await pdf(file.buffer);
        const text = pdfData.text || "";
        if (!text.trim()) continue;

        const chunks = chunkText(text, { chunkSize, chunkOverlap });

        for (let i = 0; i < chunks.length; i++) {
          const chunk = chunks[i];
          try {
            allChunks.push({
              text: chunk,
              metadata: {
                source: file.originalname,
                chunkIndex: i,
              },
            });
          } catch (error) {
            console.error(`Error generating metadata for chunk ${i}:`, error);

            allChunks.push({
              text: chunk,
              metadata: {
                source: file.originalname,
                chunkIndex: i,
              },
            });
          }
        }
      }

      if (allChunks.length === 0) {
        return res
          .status(400)
          .json({ error: "No embeddable text extracted from PDFs" });
      }

      const texts = allChunks.map((chunk) => chunk.text);
      const metadata = allChunks.map((chunk) => chunk.metadata);

      await upsertVectors(texts, metadata);

      const total = allChunks.length;
      res.json({ ok: true, upserted: total, namespace });
    } catch (err) {
      next(err);
    }
  }
);

app.post("/query", async (req: Request, res: Response, next: NextFunction) => {
  try {
    const parsed = QuerySchema.safeParse(req.body);
    if (!parsed.success) {
      return res
        .status(400)
        .json({ error: "Invalid parameters", details: parsed.error.flatten() });
    }
    const {
      query,
      topK = 5,
      namespace = "default",
      withAnswer = true,
    } = parsed.data;

    const index = await getPineconeIndex();

    console.log(`Searching for query: "${query}" in namespace: "${namespace}"`);

    // Enhanced search with higher topK for better reranking
    const searchTopK = Math.max(topK * 2, 10);

    const results = await index.searchRecords({
      query: {
        inputs: { text: query },
        topK: searchTopK,
      },
      fields: ["chunk_text", "source", "chunk_index"],
      rerank: {
        model: "bge-reranker-v2-m3",
        topN: Math.min(searchTopK, 10),
        rankFields: ["chunk_text"],
      },
    });

    console.log(`Search results:`, JSON.stringify(results, null, 2));
    const matches = results.result.hits ?? [];

    const parsedMatches = matches.map((match) => ({
      ...match,
      metadata: parseMetadata({
        source: (match.fields as any)?.source,
        chunkIndex: (match.fields as any)?.chunk_index,
      }),
      fields: match.fields || {},
    }));

    let answer: string | undefined;
    let advancedRAGResult: any = undefined;

    if (withAnswer) {
      if (parsedMatches.length > 0) {
        // Use advanced RAG techniques
        const { advancedRAGQuery } = await import("./services/advanced-rag.ts");
        try {
          advancedRAGResult = await advancedRAGQuery(
            query,
            parsedMatches,
            topK
          );
          answer = advancedRAGResult.finalAnswer;
        } catch (error) {
          console.error("Error in advanced RAG:", error);
          answer = "An error occurred while processing your request.";
        }
      } else {
        answer =
          "I don't have enough information in my knowledge base to answer this question accurately.";
      }
    }

    const response: any = {
      ok: true,
      matches: parsedMatches,
      answer,
    };

    // Include advanced RAG metadata if available
    if (advancedRAGResult) {
      response.advancedRAG = {
        expandedQueries: advancedRAGResult.expandedQueries,
        critiqueScore: advancedRAGResult.critiqueScore,
        confidence: advancedRAGResult.confidence,
        rerankedDocuments: advancedRAGResult.rerankedDocuments.length,
      };
    }

    res.json(response);
  } catch (err) {
    next(err);
  }
});

// New endpoint for advanced RAG with detailed analysis
app.post(
  "/query/advanced",
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const parsed = QuerySchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({
          error: "Invalid parameters",
          details: parsed.error.flatten(),
        });
      }
      const { query, topK = 5, namespace = "default" } = parsed.data;

      const index = await getPineconeIndex();

      console.log(
        `Advanced RAG query: "${query}" in namespace: "${namespace}"`
      );

      // Enhanced search with higher topK for better reranking
      const searchTopK = Math.max(topK * 3, 15);

      const results = await index.searchRecords({
        query: {
          inputs: { text: query },
          topK: searchTopK,
        },
        fields: ["chunk_text", "source", "chunk_index"],
        rerank: {
          model: "bge-reranker-v2-m3",
          topN: Math.min(searchTopK, 15),
          rankFields: ["chunk_text"],
        },
      });

      const matches = results.result.hits ?? [];
      const parsedMatches = matches.map((match) => ({
        ...match,
        metadata: parseMetadata({
          source: (match.fields as any)?.source,
          chunkIndex: (match.fields as any)?.chunk_index,
        }),
        fields: match.fields || {},
      }));

      if (parsedMatches.length === 0) {
        return res.json({
          ok: true,
          query,
          message: "No relevant documents found",
          advancedRAG: {
            expandedQueries: [query],
            critiqueScore: 0.0,
            confidence: 0.0,
            rerankedDocuments: [],
            documents: [],
          },
        });
      }

      // Use advanced RAG techniques
      const { advancedRAGQuery } = await import("./services/advanced-rag.ts");
      const advancedRAGResult = await advancedRAGQuery(
        query,
        parsedMatches,
        topK
      );

      res.json({
        ok: true,
        query,
        advancedRAG: advancedRAGResult,
        rawMatches: parsedMatches.slice(0, 5), // Include first 5 raw matches for comparison
      });
    } catch (err) {
      next(err);
    }
  }
);

app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  console.error(err);
  res.status(500).json({ error: "Internal server error" });
});

const port = Number(process.env.PORT || 3000);
app.listen(port, async () => {
  await ensurePineconeIndex();
  console.log(`Server listening on http://localhost:${port}`);
});
