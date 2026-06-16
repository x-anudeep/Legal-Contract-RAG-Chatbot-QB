import { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import axios from "axios";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

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

function AnswerText({ text }) {
  const paragraphs = text
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

  return (
    <div className="answer-copy">
      {paragraphs.map((paragraph, index) => {
        const isSummary = paragraph.toLowerCase().startsWith("summary:");

        return (
          <p
            className={isSummary ? "summary-paragraph" : undefined}
            key={`${paragraph.slice(0, 24)}-${index}`}
          >
            {paragraph}
          </p>
        );
      })}
    </div>
  );
}

function SourceCard({ source, index }) {
  const clauses = source.clause_categories?.length
    ? source.clause_categories.join(", ")
    : "Unlabeled";
  const rawRerankScore = Number(source.rerank_score);
  const rerankScore = Number.isFinite(rawRerankScore)
    ? rawRerankScore.toFixed(2)
    : null;
  const scoreWidth = Number.isFinite(rawRerankScore)
    ? `${Math.min(Math.max(rawRerankScore * 10, 0), 100)}%`
    : "8%";
  const scoreScale = source.rerank_score_scale || "0-10";

  return (
    <article className="source-card">
      <div className="source-header">
        <span className="source-rank">#{index + 1}</span>
        <div>
          <h3>{source.contract || source.contract_name}</h3>
          <p>{clauses}</p>
        </div>
      </div>
      {rerankScore !== null && (
        <div className="source-score" aria-label={`Rerank score ${rerankScore} out of 10`}>
          <span>Rerank score ({scoreScale})</span>
          <strong>{rerankScore}/10</strong>
          <div className="score-track">
            <span style={{ width: scoreWidth }} />
          </div>
          {source.rerank_reason && (
            <p className="score-reason">{source.rerank_reason}</p>
          )}
        </div>
      )}
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
  const topSource = sources[0];
  const topSourceTitle = topSource?.contract || topSource?.contract_name || "No source yet";
  const topSourceScore = Number(topSource?.rerank_score);
  const formattedTopScore = Number.isFinite(topSourceScore) ? `${topSourceScore.toFixed(2)}/10` : "--";

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
            <p className="subtitle">
              Ask contract questions, inspect reranked evidence, and get a plain-language summary.
            </p>
          </div>
          <span className="status-pill">Mock retriever</span>
        </header>

        <section className="dashboard-grid">
          <form className="chat-panel" onSubmit={handleSubmit}>
            <div className="panel-title-row">
              <div>
                <h2>Ask a question</h2>
                <p>Evidence is retrieved, reranked, then answered from the best clause.</p>
              </div>
              {loading && <span className="loading-dot">Searching</span>}
            </div>

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

          <aside className="insights-panel">
            <h2>Retrieval status</h2>
            <div className="metric-stack">
              <div className="metric-card">
                <span>Sources shown</span>
                <strong>{sources.length}</strong>
              </div>
              <div className="metric-card">
                <span>Top score</span>
                <strong>{formattedTopScore}</strong>
              </div>
            </div>
            <div className="top-source">
              <span>Best ranked source</span>
              <strong>{topSourceTitle}</strong>
            </div>
          </aside>
        </section>

        {error && <div className="error-banner">{error}</div>}

        {loading && (
          <section className="answer-section skeleton-panel" aria-label="Loading answer">
            <span />
            <span />
            <span />
          </section>
        )}

        {!loading && answer && (
          <section className="answer-section">
            <div className="section-heading">
              <h2>Answer</h2>
              <span>Top-ranked evidence</span>
            </div>
            <AnswerText text={answer} />
          </section>
        )}

        {sources.length > 0 && (
          <section className="sources-section">
            <div className="section-heading">
              <h2>Sources</h2>
              <span>Reranked best to weakest</span>
            </div>
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
