export interface Citation {
  doc_title: string;
  page: number;
  section: string;
  quote: string;
}

export interface ChatResponse {
  query: string;
  answer: string;
  grounded: boolean;
  citations: Citation[];
  agents_used: string[];
  sources: unknown[];
  latency_ms: number;
}

const base = "/api";

export async function chat(query: string): Promise<ChatResponse> {
  const res = await fetch(`${base}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error(`chat failed: ${res.status}`);
  return res.json();
}

export async function uploadText(title: string, text: string) {
  const res = await fetch(`${base}/upload`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, text }),
  });
  if (!res.ok) throw new Error(`upload failed: ${res.status}`);
  return res.json();
}
