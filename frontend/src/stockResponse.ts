export async function stockResponse(response: Response) {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.error || payload?.detail;
    throw new Error(typeof detail === "string" ? detail :
      `The stock service returned HTTP ${response.status}. Check that the backend is running and try again.`);
  }
  if (!payload || typeof payload !== "object")
    throw new Error("The stock service returned an invalid response. Check that the backend is running and try again.");
  return payload;
}

export function stockSearchError(reason: unknown, ticker: string, stage: string) {
  const message = reason instanceof Error ? reason.message : "";
  if (/No downloaded price history|No local data/.test(message))
    return `No saved price data for ${ticker}. Try another stock symbol.`;
  if (/Not enough|Need more than/.test(message))
    return `There is not enough completed price history to train ${ticker} yet.`;
  if (/integrity|Checksum|do not match/.test(message))
    return "Saved results could not be verified. No figures have been displayed.";
  if (/Failed to fetch|NetworkError|fetch failed/i.test(message))
    return `Could not ${stage} for ${ticker}. Check your connection and that the backend is running, then try again.`;
  return `Could not ${stage} for ${ticker}. ${message || "Please try again."}`;
}
