import {describe, expect, it} from 'vitest';

import {resolveDevServerConfig} from '../../devServerConfig';

describe('resolveDevServerConfig', () => {
  it('uses values loaded from the Vite env file', () => {
    expect(
      resolveDevServerConfig({
        VITE_DEV_PORT: '4317',
        DEV_API_PROXY_TARGET: ' http://127.0.0.1:8123 ',
      }),
    ).toEqual({port: 4317, apiProxyTarget: 'http://127.0.0.1:8123'});
  });

  it('falls back safely for an invalid port and empty proxy target', () => {
    expect(
      resolveDevServerConfig({VITE_DEV_PORT: 'not-a-port', DEV_API_PROXY_TARGET: ' '}),
    ).toEqual({port: 3000, apiProxyTarget: 'http://localhost:8000'});
  });
});
