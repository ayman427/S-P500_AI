export type Stock = { ticker: string; name: string };

const normalized = (value: string) => value.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]/g, "");
const aliases: Record<string, string[]> = {
  google: ["GOOGL", "GOOG"], facebook: ["META"], meta: ["META"],
};

export function stockMatches(query: string, stocks: Stock[]): Stock[] {
  const text = query.trim();
  const ticker = text.toUpperCase().replaceAll(".", "-");
  const explicit = text.match(/\(([A-Za-z0-9.-]+)\)\s*$/)?.[1].toUpperCase().replaceAll(".", "-");
  const exact = stocks.find((stock) => stock.ticker === (explicit ?? ticker));
  if (exact) return [exact];
  const key = normalized(text);
  if (!key) return [];
  const aliasTickers = aliases[key] ?? [];
  const exactNames = stocks.filter((stock) => normalized(stock.name) === key || aliasTickers.includes(stock.ticker));
  if (exactNames.length) return exactNames;
  return stocks.filter((stock) => normalized(stock.name).includes(key) || normalized(stock.ticker).startsWith(key));
}

export function resolveStock(query: string, stocks: Stock[]): Stock[] {
  const matches = stockMatches(query, stocks);
  if (matches.length) return matches;
  // Let the API explain unavailable symbols, but never guess a company-name match.
  if (/^[A-Z][A-Z0-9.-]{0,14}$/.test(query.trim())) {
    const ticker = query.trim().replaceAll(".", "-");
    return [{ ticker, name: ticker }];
  }
  return [];
}
