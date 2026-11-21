# Northstar Cloud | Engineering Runbook (DEMO DATA)
Northstar Cloud is a fictional SaaS company for illustrating document-grounded RAG.

## Deployment procedure
All production deployments require a passing CI pipeline, code review from one peer, and a rollback plan. Canary deploy to 10 percent of traffic for 20 minutes before full rollout.

## Infrastructure
The service runs Python FastAPI containers and PostgreSQL on a managed cloud platform. Redis is used for short-lived caching and BullMQ powers asynchronous analytics workflows.

## Incident diagnosis
Check application error rate, p95 latency, and database connection saturation. For elevated latency inspect slow queries, cache hit rate, and container CPU usage.

## Recovery
Rollback to the last healthy container image when user-visible failures exceed the alert threshold for 10 minutes.
