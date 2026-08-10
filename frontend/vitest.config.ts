import {defineConfig} from 'vitest/config';

export default defineConfig({
  define: {
    __APP_VERSION__: JSON.stringify('2.5.0-test'),
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    restoreMocks: true,
  },
});
