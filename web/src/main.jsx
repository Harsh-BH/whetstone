import React from "react";
import ReactDOM from "react-dom/client";
import SkillRadar from "./Radar.jsx";
import DailySet from "./DailySet.jsx";

function App() {
  return (
    <div style={{ fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ textAlign: "center", marginTop: 24 }}>Whetstone</h1>
      <SkillRadar />
      <DailySet />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
