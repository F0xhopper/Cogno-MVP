import { Pinecone, Index } from "@pinecone-database/pinecone";

const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || "cogno-rag";
const PINECONE_CLOUD = process.env.PINECONE_CLOUD || "aws";
const PINECONE_REGION = process.env.PINECONE_REGION || "us-east-1";

const EMBEDDING_DIMENSION = Number(process.env.EMBEDDING_DIMENSION || 1536);

let pineconeClient: Pinecone | null = null;
let pineconeIndex: Index | null = null;

function getClient(): Pinecone {
  if (!pineconeClient) {
    const apiKey = process.env.PINECONE_API_KEY;
    if (!apiKey) {
      throw new Error("Missing PINECONE_API_KEY");
    }
    pineconeClient = new Pinecone({ apiKey });
  }
  return pineconeClient;
}

export async function ensurePineconeIndex(): Promise<Index> {
  if (pineconeIndex) return pineconeIndex;

  const pc = getClient();

  const indexes = await pc.listIndexes();
  const hasIndex =
    indexes.indexes?.some((idx: any) => idx.name === PINECONE_INDEX_NAME) ||
    false;

  if (!hasIndex) {
    await pc.createIndexForModel({
      name: PINECONE_INDEX_NAME,
      cloud: PINECONE_CLOUD as "aws" | "gcp",
      region: PINECONE_REGION,
      embed: {
        model: "llama-text-embed-v2",
        fieldMap: { text: "chunk_text" },
      },
    });

    await new Promise((resolve) => setTimeout(resolve, 10000));
  }

  pineconeIndex = pc.Index(PINECONE_INDEX_NAME);
  return pineconeIndex;
}

export async function getPineconeIndex(): Promise<Index> {
  if (pineconeIndex) return pineconeIndex;
  return ensurePineconeIndex();
}

export async function upsertVectors(texts: string[], metadata: any[]) {
  const index = await getPineconeIndex();
  const BATCH_SIZE = 96;

  const records = texts.map((text, i) => ({
    id: `chunk_${Date.now()}_${i}`,
    values: [],
    chunk_text: text,
    metadata: {
      ...metadata[i],
    },
  }));

  const results = [];
  for (let i = 0; i < records.length; i += BATCH_SIZE) {
    const batch = records.slice(i, i + BATCH_SIZE);
    console.log(
      `Upserting batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(
        records.length / BATCH_SIZE
      )} (${batch.length} records)`
    );

    const result = await index.upsertRecords(batch);
    results.push(result);
  }

  return results;
}

export const PINECONE_INDEX = PINECONE_INDEX_NAME;
export const EMBED_DIM = EMBEDDING_DIMENSION;
