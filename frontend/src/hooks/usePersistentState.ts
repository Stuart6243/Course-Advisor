import {useCallback, useEffect, useState} from 'react';

/**
 * 带 localStorage 持久化的 state。
 *
 * 旧版语言和查询设置都只存在内存里，刷新页面就回到英文 / 10 轮 / 5 条，
 * 用户在设置面板里改完再刷新一下就白改了。
 *
 * localStorage 在部分环境下会抛异常（隐私模式、iframe 限制），
 * 因此所有访问都包了 try/catch，失败时静默退化为普通 useState。
 */
export function usePersistentState<T>(
  key: string,
  defaultValue: T,
  validate?: (value: unknown) => value is T,
): [T, (value: T) => void] {
  const [state, setState] = useState<T>(() => {
    try {
      const raw = window.localStorage.getItem(key);
      if (raw === null) {
        return defaultValue;
      }
      const parsed = JSON.parse(raw) as unknown;
      if (validate && !validate(parsed)) {
        return defaultValue;
      }
      return parsed as T;
    } catch {
      return defaultValue;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(state));
    } catch {
      // 存不进去也不影响本次会话使用
    }
  }, [key, state]);

  const set = useCallback((value: T) => setState(value), []);

  return [state, set];
}
