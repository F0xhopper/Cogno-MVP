export function chunkText(
  text: string,
  options?: { chunkSize?: number; chunkOverlap?: number }
): string[] {
  const chunkSize = options?.chunkSize ?? 1200;
  const chunkOverlap = options?.chunkOverlap ?? 200;

  const normalized = text
    .replace(/\r\n/g, "\n")
    .replace(/\t/g, " ")
    .replace(/[\u0000-\u001F\u007F]+/g, " ")
    .replace(/\s+$/gm, "")
    .trim();

  const words = normalized.split(/\s+/);
  const chunks: string[] = [];
  let current: string[] = [];
  let currentLength = 0;

  for (const word of words) {
    const len = word.length + 1;
    if (currentLength + len > chunkSize && current.length > 0) {
      chunks.push(current.join(" "));

      if (chunkOverlap > 0) {
        const overlapWords = Math.max(0, Math.floor(chunkOverlap / 6));
        current = current.slice(-overlapWords);
        currentLength = current.reduce((acc, w) => acc + w.length + 1, 0);
      } else {
        current = [];
        currentLength = 0;
      }
    }
    current.push(word);
    currentLength += len;
  }
  if (current.length) chunks.push(current.join(" "));
  return chunks;
}
