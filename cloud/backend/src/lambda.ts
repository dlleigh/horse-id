import serverlessExpress from "@codegenie/serverless-express";
import { ensureConfig } from "./services/config.js";
import app from "./app.js";

const se = serverlessExpress({ app });

// Load SSM params once on first invocation
const ready = ensureConfig();

export const handler = async (event: Record<string, unknown>, context: unknown) => {
  await ready;

  // Lambda Function URL events look like API Gateway v2 but
  // serverless-express doesn't auto-detect them. Force the event source.
  if (event.requestContext && (event.requestContext as Record<string, unknown>).http) {
    (event as Record<string, unknown>).version = "2.0";
  }
  return se(event, context);
};
