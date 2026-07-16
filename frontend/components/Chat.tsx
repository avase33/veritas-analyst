"use client";

import { useEffect, useRef, useState } from "react";
import { chat, type Citation } from "@/lib/api";

interface Message {
  role: "user" | "analyst";
  text: string;
  grounded?: boolean;
  citations?: Citation[];
  agents?: string[];
}

const SAMPLES = [
  "What was the total revenue growth in Q3, and what are the primary risk factors?",
  "What is the limitation of liability in the services agreement?",
  "Who is the current CEO and what is their salary?",
];

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send(text?: string) {
    const query = (text ?? q).trim();
    if (!query || busy) return;
    setMessages((m) => [...m, { role: "user", text: query }]);
    setQ("");
    setBusy(true);
    try {
      const d = await chat(query);
      setMessages((m) => [...m, {
        role: "analyst", text: d.answer, grounded: d.grounded,
        citations: d.citations, agents: d.agents_used,
      }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "analyst", text: `Error: ${err}` }]);
    }
    setBusy(false);
  }

  return (
    <div className="wrap">
      <div className="hint">
        Two sample documents are loaded (annual report + services agreement). Try:{" "}
        {SAMPLES.map((s, i) => (
          <span key={i}>
            {i > 0 ? " · " : ""}
            <b onClick={() => void send(s)}>&ldquo;{s.slice(0, 38)}…&rdquo;</b>
          </span>
        ))}
      </div>
      <div className="msgs">
        {messages.map((m, i) => (
          <div className="msg" key={i}>
            <div className="role">{m.role}</div>
            <div className={"bubble" + (m.role === "user" ? " user" : "") + (m.grounded === false ? " refused" : "")}>
              {m.text}
              {m.citations && m.citations.length > 0 && (
                <div className="cites">
                  Sources:{" "}
                  {m.citations.map((c, j) => (
                    <span className="cite" key={j}>
                      {c.doc_title ? `${c.doc_title} · ` : ""}Page {c.page}
                      {c.section ? ` · ${c.section}` : ""}
                    </span>
                  ))}
                </div>
              )}
              {m.agents && m.agents.length > 0 && (
                <div className="agents">
                  {m.agents.map((a, j) => (
                    <span className="agent" key={j}>{a}</span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {busy && (
          <div className="msg">
            <div className="role">analyst</div>
            <div className="bubble">Thinking…</div>
          </div>
        )}
        <div ref={end} />
      </div>
      <form onSubmit={(e) => { e.preventDefault(); void send(); }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about the documents…" />
        <button type="submit">Ask</button>
      </form>
    </div>
  );
}
