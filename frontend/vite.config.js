import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';

// Use environment variable or default localhost
const backendURL = process.env.BACKEND_URL || 'http://114.79.161.52:9900';

export default defineConfig({
  plugins: [react()],
  server: {
    host:'0.0.0.0',
    port: 5173,
    https:{
      key: fs.readFileSync(path.resolve(__dirname, 'certs/key.pem')),
      cert: fs.readFileSync(path.resolve(__dirname, 'certs/cert.pem'))
    },
    proxy: {
      '/ws': {
        target: backendURL.replace(/^http/, 'ws'), // WebSocket proxy
        ws: true
      }
    }
  },
  define: {
    'process.env':{}
  }
});
