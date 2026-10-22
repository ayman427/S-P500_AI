/* eslint-disable */
/**
 * Generated `api` utility.
 *
 * THIS CODE IS AUTOMATICALLY GENERATED.
 *
 * To regenerate, run `npx convex dev`.
 * @module
 */

import type * as datasets from "../datasets.js";
import type * as insiderTransactions from "../insiderTransactions.js";
import type * as marketWatch from "../marketWatch.js";
import type * as predictions from "../predictions.js";
import type * as spikeResearch from "../spikeResearch.js";
import type * as spikeWatch from "../spikeWatch.js";

import type {
  ApiFromModules,
  FilterApi,
  FunctionReference,
} from "convex/server";

declare const fullApi: ApiFromModules<{
  datasets: typeof datasets;
  insiderTransactions: typeof insiderTransactions;
  marketWatch: typeof marketWatch;
  predictions: typeof predictions;
  spikeResearch: typeof spikeResearch;
  spikeWatch: typeof spikeWatch;
}>;

/**
 * A utility for referencing Convex functions in your app's public API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = api.myModule.myFunction;
 * ```
 */
export declare const api: FilterApi<
  typeof fullApi,
  FunctionReference<any, "public">
>;

/**
 * A utility for referencing Convex functions in your app's internal API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = internal.myModule.myFunction;
 * ```
 */
export declare const internal: FilterApi<
  typeof fullApi,
  FunctionReference<any, "internal">
>;

export declare const components: {};
