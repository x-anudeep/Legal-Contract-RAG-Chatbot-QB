import { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import axios from "axios";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const CLAUSE_CATEGORIES = [
  "Document Name",
  "Parties",
  "Agreement Date",
  "Effective Date",
  "Expiration Date",
  "Renewal Term",
  "Notice to Terminate Renewal",
  "Governing Law",
  "Most Favored Nation",
  "Non-Compete",
  "Exclusivity",
  "No-Solicit of Customers",
  "No-Solicit of Employees",
  "Non-Disparagement",
  "Termination for Convenience",
  "ROFR/ROFO/ROFN",
  "Change of Control",
  "Anti-Assignment",
  "Revenue/Profit Sharing",
  "Price Restriction",
  "Minimum Commitment",
  "Volume Restriction",
  "IP Ownership Assignment",
  "Joint IP Ownership",
  "License Grant",
  "Non-Transferable License",
  "Affiliate IP License",
  "Unlimited/All-You-Can-Eat License",
  "Irrevocable or Perpetual License",
  "Source Code Escrow",
  "Post-Agreement Restrictions",
  "Audit Rights",
  "Uncapped Liability",
  "Cap On Liability",
  "Liquidated Damages",
  "Warranty Duration",
  "Insurance",
  "Covenant Not to Sue",
  "Third Party Beneficiary",
  "Indemnification",
  "Dispute Resolution",
];

const EXAMPLE_QUESTIONS = [
  "What is the governing law?",
  "Can either party terminate without cause?",
  "Is the license non-transferable?",
  "Does this contract include a cap on liability?",
];

function SourceCard({ source, index }) {
  const clauses = source.clause_categories?.length
    ? source.clause_categories.join(", ")
    : "Unlabeled";

  return (
    <article className="source-card">
      <div className="source-header">
        <span className="source-rank">#{index + 1}</span>
        <div>
          <h3>{source.contract || source.contract_name}</h3>
          <p>{clauses}</p>
        </div>
      </div>
      <p className="source-preview">{source.text_preview}</p>
    </article>
  );
}

function App() {
  const [question, setQuestion] = useState("");
  const [clauseFilter, setClauseFilter] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const canSubmit = useMemo(() => question.trim().length > 0 && !loading, [question, loading]);

  async function handleSubmit(event) {
    event?.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        question: question.trim(),
        clause_filter: clauseFilter || null,
      });
      setAnswer(response.data.answer);
      setSources(response.data.sources || []);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">CUAD RAG</p>
            <h1>Contract Review Chatbot</h1>
          </div>
          <span className="status-pill">Mock retriever</span>
        </header>

        <form className="chat-panel" onSubmit={handleSubmit}>
          <label className="field-label" htmlFor="clause-filter">
            Clause category
          </label>
          <select
            id="clause-filter"
            value={clauseFilter}
            onChange={(event) => setClauseFilter(event.target.value)}
          >
            <option value="">All clause types</option>
            {CLAUSE_CATEGORIES.map((category) => (
              <option key={category} value={category}>
                {category}
              </option>
            ))}
          </select>

          <label className="field-label" htmlFor="question">
            Question
          </label>
          <div className="question-row">
            <input
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about governing law, termination, liability, license terms..."
            />
            <button type="submit" disabled={!canSubmit}>
              {loading ? "Searching" : "Ask"}
            </button>
          </div>

          <div className="examples">
            {EXAMPLE_QUESTIONS.map((example) => (
              <button
                type="button"
                key={example}
                onClick={() => setQuestion(example)}
              >
                {example}
              </button>
            ))}
          </div>
        </form>

        {error && <div className="error-banner">{error}</div>}

        {answer && (
          <section className="answer-section">
            <h2>Answer</h2>
            <p>{answer}</p>
          </section>
        )}

        {sources.length > 0 && (
          <section className="sources-section">
            <h2>Sources</h2>
            <div className="source-grid">
              {sources.map((source, index) => (
                <SourceCard
                  key={`${source.contract || source.contract_name}-${source.chunk_id ?? index}`}
                  source={source}
                  index={index}
                />
              ))}
            </div>
          </section>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
