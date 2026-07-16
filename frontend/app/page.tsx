import { Chat } from "@/components/Chat";

export default function Page() {
  return (
    <>
      <header>
        <h1>Veritas</h1>
        <span className="badge">multi-agent RAG · grounded answers</span>
      </header>
      <Chat />
    </>
  );
}
