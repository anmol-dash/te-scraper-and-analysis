import path from 'node:path';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

function tauriDevHost(): string | false {
  const raw = process.env.TAURI_DEV_HOST?.trim();
  return raw ? raw : false;
}

const host = tauriDevHost();

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  envPrefix: ['VITE_', 'TAURI_'],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: 'ws',
          host,
          port: 1420,
        }
      : undefined,
    watch: {
      ignored: ['**/src-tauri/**'],
    },
  },
});
