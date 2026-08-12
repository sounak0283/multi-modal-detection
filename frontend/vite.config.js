import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Built output lands in ../backend/web/dist, which FastAPI serves in production.
// Building INTO the backend keeps a deployment to "one process, one port" - the backend
// directory is a self-contained deployable unit, with no separate web server to install
// and secure on a customer's box.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: '../backend/web/dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    // During development Vite serves the UI and forwards API calls to the Python
    // process, so the browser sees a single origin and there is no CORS to configure.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
