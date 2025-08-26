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
import { generateChunkMetadata } from "./services/rag.ts";
import pdf from "pdf-parse";

const app = express();
app.use(express.json({ limit: "10mb" }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
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
      console.log("Request body:", body);
      console.log("Request files:", req.files);

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
            const aiMetadata = await generateChunkMetadata(chunk);

            allChunks.push({
              text: chunk,
              metadata: {
                source: file.originalname,
                chunkIndex: i,

                summary: aiMetadata.summary,
                topics: aiMetadata.topics,
                entities: aiMetadata.entities,
                documentType: aiMetadata.documentType,
                keyPoints: aiMetadata.keyPoints,
                sentiment: aiMetadata.sentiment,
                complexity: aiMetadata.complexity,
              },
            });
          } catch (error) {
            console.error(`Error generating metadata for chunk ${i}:`, error);

            allChunks.push({
              text: chunk,
              metadata: {
                source: file.originalname,
                chunkIndex: i,
                topics: ["general"],
                entities: [],
                documentType: "other",
                keyPoints: [chunk.substring(0, 100) + "..."],
                sentiment: "neutral",
                complexity: "moderate",
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

    const results = await index.searchRecords({
      query: {
        inputs: { text: query },
        topK,
      },
      fields: ["chunk_text", "metadata"],
      rerank: {
        model: "bge-reranker-v2-m3",
        topN: Math.min(topK, 5),
        rankFields: [
          "chunk_text",
          "metadata.summary",
          "metadata.topics",
          "metadata.entities",
        ],
      },
    });

    const matches = results.result.hits ?? [];

    let answer: string | undefined;
    if (withAnswer && matches.length > 0) {
      const contexts = matches
        .map((m) => (m.fields as any).chunk_text)
        .filter(Boolean)
        .slice(0, 10);

      const prompt = `You are a helpful assistant. Using only the context below, answer the user question.\n\nContext:\n${contexts
        .map((c, i) => `[${i + 1}] ${c ?? ""}`)
        .join("\n\n")}\n\nQuestion: ${query}\n\nAnswer:`;

      const { getChatCompletion } = await import("./services/rag.ts");
      answer = await getChatCompletion(prompt);
    }

    res.json({ ok: true, matches, answer });
  } catch (err) {
    next(err);
  }
});

app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  console.error(err);
  res.status(500).json({ error: "Internal server error" });
});

const port = Number(process.env.PORT || 3000);
app.listen(port, async () => {
  await ensurePineconeIndex();
  console.log(`Server listening on http://localhost:${port}`);
});
