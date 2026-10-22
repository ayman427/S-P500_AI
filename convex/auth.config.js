// The issuer is configured per deployment. No issuer means no accepted users.
export default {
  providers: process.env.CLERK_JWT_ISSUER_DOMAIN ? [{
    domain: process.env.CLERK_JWT_ISSUER_DOMAIN,
    applicationID: "convex",
  }] : [],
};
