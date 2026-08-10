export function resolveDevServerConfig(env: Record<string, string | undefined>) {
  const requestedPort = Number(env.VITE_DEV_PORT ?? 3000);
  return {
    port: Number.isInteger(requestedPort) && requestedPort > 0 ? requestedPort : 3000,
    apiProxyTarget:
      env.DEV_API_PROXY_TARGET?.trim() ||
      env.VITE_API_PROXY_TARGET?.trim() ||
      'http://localhost:8000',
  };
}
