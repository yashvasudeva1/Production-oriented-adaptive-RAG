import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/ready': 'http://127.0.0.1:8000',
      '/metrics': 'http://127.0.0.1:8000',
      '/query': 'http://127.0.0.1:8000',
      '/search': 'http://127.0.0.1:8000',
      '/documents': 'http://127.0.0.1:8000',
    },
  },
});
