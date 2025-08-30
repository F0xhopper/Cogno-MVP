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