import { createInternalNeonAuth } from "@neondatabase/auth";
import { BetterAuthReactAdapter } from "@neondatabase/auth/react/adapters";

const NEON_AUTH_URL = import.meta.env.VITE_NEON_AUTH_URL as string;

const neonAuth = createInternalNeonAuth(NEON_AUTH_URL, {
  adapter: BetterAuthReactAdapter(),
});

export const authClient = neonAuth.adapter;
export const getJWTToken = () => neonAuth.getJWTToken();
