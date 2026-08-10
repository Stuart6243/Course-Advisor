import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';

import pkg from './package.json' with {type: 'json'};
import {resolveDevServerConfig} from './devServerConfig';

export default defineConfig(({mode}) => {
  // Vite loads .env files after config evaluation unless loadEnv is called here.
  const env = loadEnv(mode, __dirname, '');
  const {port, apiProxyTarget} = resolveDevServerConfig(env);

  return {
    define: {
      __APP_VERSION__: JSON.stringify(pkg.version),
    },
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      port,
      host: '0.0.0.0',
      proxy: {
        '/api': {
          target: apiProxyTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
