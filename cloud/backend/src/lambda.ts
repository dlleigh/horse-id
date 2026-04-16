import serverlessExpress from "@codegenie/serverless-express";
import app from "./app.js";

const se = serverlessExpress({ app });

export const handler = (event: Record<string, unknown>, context: unknown) => {
  // Lambda Function URL events look like API Gateway v2 but
  // serverless-express doesn't auto-detect them. Force the event source.
  if (event.requestContext && (event.requestContext as Record<string, unknown>).http) {
    (event as Record<string, unknown>).version = "2.0";
  }
  return se(event, context);
};
