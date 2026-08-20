import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// self-hosted variable faces; globals.css names them in --font-ui/serif/mono
import "@fontsource-variable/outfit";
import "@fontsource-variable/source-serif-4";
import "@fontsource-variable/jetbrains-mono";

import "@/globals.css";
import App from "@/App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
