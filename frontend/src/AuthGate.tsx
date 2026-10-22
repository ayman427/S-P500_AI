import { SignIn, SignOutButton, useAuth } from "@clerk/react";
import { useEffect, useState } from "react";
import App from "./App";
import { useAuthenticatedFetch } from "./useAuthenticatedFetch";

function VerifiedSession({ userId }: { userId: string }) {
  const apiFetch = useAuthenticatedFetch();
  const [verification, setVerification] = useState<{ ready: boolean; error?: string }>({ ready: false });
  useEffect(() => {
    const controller = new AbortController();
    apiFetch("/api/auth/me", { signal: controller.signal }).then(async response => {
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || payload.detail || "Could not verify sign-in.");
      if (payload.userId !== userId) throw new Error("The signed-in account could not be verified.");
      if (!controller.signal.aborted) setVerification({ ready: true });
    }).catch(error => {
      if (!controller.signal.aborted) setVerification({ ready: false, error: error instanceof Error ? error.message : "Could not verify sign-in." });
    });
    return () => controller.abort();
  }, [apiFetch, userId]);
  if (verification.error) return <main className="auth-page"><p role="alert">{verification.error}</p><button type="button" onClick={() => window.location.reload()}>Try again</button><SignOutButton><button type="button">Sign out</button></SignOutButton></main>;
  if (!verification.ready) return <main className="auth-page"><p role="status">Verifying your session…</p></main>;
  return <App userId={userId} />;
}

export default function AuthGate() {
  const { isLoaded, isSignedIn, userId, sessionId } = useAuth();
  if (!isLoaded) return <main className="auth-page"><p role="status">Loading your session…</p></main>;
  if (!isSignedIn) return (
    <main className="auth-page">
      <div className="auth-intro"><h1>Welcome</h1><p>Sign in to search stocks and keep your prediction history private.</p></div>
      <SignIn routing="hash" forceRedirectUrl="/" />
    </main>
  );
  // A new user/session gets fresh component state and aborts the old user's requests.
  return <VerifiedSession key={`${userId}:${sessionId}`} userId={userId!} />;
}
