import { useEffect, useState } from "react";

const API = "http://localhost:8000";
const USER = "Vish2503"; // demo validation handle; real user plugs in here

const MODE_COLOR = { assess: "#d97706", train: "#2563eb" };

export default function DailySet() {
  const [items, setItems] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/daily-set?user=${USER}`)
      .then((r) => r.json())
      .then(setItems)
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <p style={{ color: "#b91c1c" }}>API error: {err} (is `make serve` running?)</p>;
  if (!items) return <p>Loading today's set…</p>;

  return (
    <div style={{ maxWidth: 760, margin: "0 auto", padding: 16 }}>
      <h2>Today's set — {USER}</h2>
      {items.map((it) => (
        <div key={it.pid} style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 12, margin: "8px 0" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <a href={it.url} target="_blank" rel="noreferrer" style={{ fontWeight: 600 }}>
              {it.pid} (rating {it.b})
            </a>
            <span style={{ background: MODE_COLOR[it.mode] || "#6b7280", color: "white",
                           borderRadius: 6, padding: "2px 8px", fontSize: 12 }}>
              {it.mode} · P≈{Math.round(it.predicted_p * 100)}%
            </span>
          </div>
          <div style={{ color: "#6b7280", fontSize: 13, marginTop: 4 }}>{it.tags.join(", ")}</div>
          <div style={{ fontSize: 14, marginTop: 6 }}>{it.why}</div>
        </div>
      ))}
    </div>
  );
}
