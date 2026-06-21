import { useEffect, useState } from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Legend,
  ResponsiveContainer,
} from "recharts";

const TARGET_R = 1900; // docs/01 default target rating

export default function SkillRadar() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch("/skills.json")
      .then((r) => r.json())
      .then((rows) =>
        setData(
          rows.map((d) => ({
            tag: d.tag,
            mu: Math.round(d.mu),
            low: Math.round(d.mu - d.sigma), // conservative (μ−σ) band
          }))
        )
      )
      .catch(() => setData([]));
  }, []);

  return (
    <div style={{ width: "100%", maxWidth: 1000, margin: "0 auto", padding: 16 }}>
      <h2 style={{ textAlign: "center" }}>Skill radar — μ per tag (μ−σ band)</h2>
      <p style={{ textAlign: "center", color: "#6b7280" }}>
        Target R = {TARGET_R}. Inner grey ring is the conservative μ−σ estimate.
      </p>
      <div style={{ width: "100%", height: 620 }}>
        <ResponsiveContainer>
          <RadarChart data={data} outerRadius="78%">
            <PolarGrid />
            <PolarAngleAxis dataKey="tag" tick={{ fontSize: 11 }} />
            <PolarRadiusAxis domain={[800, 3000]} tick={{ fontSize: 10 }} />
            <Radar name="μ (skill)" dataKey="mu" stroke="#2563eb" fill="#2563eb" fillOpacity={0.4} />
            <Radar name="μ−σ" dataKey="low" stroke="#9ca3af" fill="#9ca3af" fillOpacity={0.2} />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
