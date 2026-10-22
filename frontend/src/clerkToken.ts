type TokenOptions = { skipCache?: boolean; template?: string };
type GetToken = (options?: TokenOptions) => Promise<string | null>;

function hasConvexAudience(token: string) {
  // Only select the Clerk token type here; Python and Convex verify authenticity.
  try {
    const encoded = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const { aud } = JSON.parse(atob(encoded));
    return aud === "convex" || (Array.isArray(aud) && aud.includes("convex"));
  } catch { return false; }
}

export async function getConvexToken(getToken: GetToken, forceRefresh = false) {
  let token = await getToken({ skipCache: forceRefresh });
  if (!token) throw new Error("Please sign in again to continue.");
  if (hasConvexAudience(token)) return token;
  // An integration enabled after sign-in may leave a cached token without aud.
  if (!forceRefresh) {
    token = await getToken({ skipCache: true });
    if (!token) throw new Error("Please sign in again to continue.");
    if (hasConvexAudience(token)) return token;
  }
  try {
    token = await getToken({ template: "convex", skipCache: true });
  } catch {
    throw new Error("Clerk did not provide a Convex token. Check that its Convex integration is enabled, then sign out and sign in again.");
  }
  if (!token || !hasConvexAudience(token))
    throw new Error("Clerk authentication needs a token with the Convex audience. Check the Convex integration settings.");
  return token;
}
