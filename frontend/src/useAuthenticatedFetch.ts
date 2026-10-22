import { useCallback } from "react";
import { useAuth } from "@clerk/react";
import { getConvexToken } from "./clerkToken";

export function useAuthenticatedFetch() {
  const { getToken } = useAuth();
  return useCallback(async (url: string, options: RequestInit = {}) => {
    const send = async (forceRefresh: boolean) => {
      const token = await getConvexToken(getToken, forceRefresh);
      const headers = new Headers(options.headers);
      headers.set("Authorization", `Bearer ${token}`);
      return fetch(url, { ...options, headers, cache: "no-store" });
    };
    let response = await send(false);
    // A 401 is returned before API work; refreshing once is safe for POSTs too.
    if (response.status === 401 && !options.signal?.aborted) response = await send(true);
    if (response.status === 401) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || payload.detail || "Your sign-in session could not be verified. Please sign in again.");
    }
    return response;
  }, [getToken]);
}
