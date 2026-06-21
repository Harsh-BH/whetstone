import { useEffect, useState } from "react";
import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const API = "http://localhost:8000";
const USER = "Vish2503";

function useJson(path) {
  const [data, setData] = useState(null);
  useEffect(() => {
    fetch(`${API}${path}`).then((r) => r.json()).then(setData).catch(() => setData(null));
  }, [path]);
  return data;
}

export default function Dashboard() {
  const readiness = useJson(`/readiness?user=${USER}`);
  const reviews = useJson(`/reviews?user=${USER}`);
  const mastery = useJson(`/mastery?user=${USER}`);
  const ratings = useJson(`/rating-history?user=${USER}`);

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 16 }}>
      <h2>Dashboard — {USER}</h2>

      {readiness && (
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 12, marginBottom: 12 }}>
          <b>Contest readiness</b> (target {readiness.target}):{" "}
          <span style={{ fontSize: 20 }}>{readiness.readiness} → {readiness.projected}</span>
          <div style={{ color: "#6b7280", fontSize: 13 }}>
            solve these {readiness.do_these.length}: {readiness.do_these.join(", ")}
          </div>
        </div>
      )}

      {ratings && ratings.actual.length > 0 && (
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 12, marginBottom: 12 }}>
          <b>Rating: actual vs predicted ({ratings.predicted})</b>
          <div style={{ width: "100%", height: 220 }}>
            <ResponsiveContainer>
              <LineChart data={ratings.actual.map((d, i) => ({ i, rating: d.rating }))}>
                <XAxis dataKey="i" tick={{ fontSize: 10 }} />
                <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Line dataKey="rating" stroke="#2563eb" dot={false} />
                <ReferenceLine y={ratings.predicted} stroke="#d97706" strokeDasharray="4 4" label="predicted" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 280, border: "1px solid #e5e7eb", borderRadius: 8, padding: 12 }}>
          <b>Due for review</b> ({reviews ? reviews.length : "…"})
          {reviews && reviews.map((d) => (
            <div key={d.concept} style={{ fontSize: 13, display: "flex", justifyContent: "space-between" }}>
              <span>{d.concept}</span>
              <span style={{ color: "#b91c1c" }}>R={d.retrievability}</span>
            </div>
          ))}
        </div>
        <div style={{ flex: 1, minWidth: 280, border: "1px solid #e5e7eb", borderRadius: 8, padding: 12 }}>
          <b>Mastery</b>
          {mastery && mastery.map((m) => (
            <div key={m.tag} style={{ fontSize: 13, display: "flex", justifyContent: "space-between" }}>
              <span>{m.mastered ? "✅" : "▫️"} {m.tag}</span>
              <span style={{ color: "#6b7280" }}>μ={m.mu}±{m.sigma}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
