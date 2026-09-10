import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/login': 'http://localhost:5004',
      '/register': 'http://localhost:5004',
      '/portfolio': 'http://localhost:5004',
      '/logout': 'http://localhost:5004',
      '/static': 'http://localhost:5004',
    },
  },
})
