import OpenAI from "openai";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  throw new Error("Missing OPENAI_API_KEY");
}

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const CHAT_MODEL = process.env.GPT_MODEL || "gpt-4o-mini";

export async function getChatCompletion(prompt: string): Promise<string> {
  const completion = await openai.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      {
        role: "system",
        content: "You are a concise assistant. If unsure, say you do not know.",
      },
      { role: "user", content: prompt },
    ],
    temperature: 0.2,
  });
  return completion.choices[0]?.message?.content?.trim() ?? "";
}

export async function generateChunkMetadata(text: string): Promise<{
  summary: string;
  topics: string[];
  entities: string[];
  documentType: string;
  keyPoints: string[];
  sentiment: "positive" | "negative" | "neutral";
  complexity: "simple" | "moderate" | "complex";
}> {
  const prompt = `Analyze the following text and extract structured metadata. Return ONLY a valid JSON object with these exact fields:

{
  "summary": "2-3 sentence summary of the main content",
  "topics": ["array of 3-5 main topics or themes"],
  "entities": ["array of 3-7 key entities like people, organizations, places, dates"],
  "documentType": "one of: technical, legal, academic, business, creative, news, other",
  "keyPoints": ["array of 3-5 key points or insights"],
  "sentiment": "one of: positive, negative, neutral",
  "complexity": "one of: simple, moderate, complex"
}

Text to analyze:
${text}

JSON:`;

  try {
    const response = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        {
          role: "system",
          content:
            "You are a metadata extraction specialist. Always return valid JSON.",
        },
        { role: "user", content: prompt },
      ],
      temperature: 0.1,
      response_format: { type: "json_object" },
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error("No response from OpenAI");
    }

    return JSON.parse(content);
  } catch (error) {
    console.error("Error generating metadata:", error);
    return {
      summary: text.substring(0, 150) + "...",
      topics: ["general"],
      entities: [],
      documentType: "other",
      keyPoints: [text.substring(0, 100) + "..."],
      sentiment: "neutral",
      complexity: "moderate",
    };
  }
}
