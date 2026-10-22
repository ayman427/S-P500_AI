import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Reuse the server's exact trusted origins for tunnel host validation.
  // Reading this value in the config does not expose server secrets to clients.
  const env = loadEnv(mode, "..", "CLERK_AUTHORIZED_PARTIES");
  const origins = process.env.CLERK_AUTHORIZED_PARTIES ?? env.CLERK_AUTHORIZED_PARTIES ?? "";
  const allowedHosts = origins.split(",").filter(origin => origin.trim()).map(origin => new URL(origin.trim()).hostname);
  return {
  envDir: "..",
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    allowedHosts,
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
  };
});
