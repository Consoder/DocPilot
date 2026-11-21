# Northstar Cloud | Security Handbook (DEMO DATA)
Northstar Cloud is a fictional SaaS company for illustrating document-grounded RAG.

## Authentication
Employees must enable multifactor authentication (MFA) for all company accounts. Production API keys must be placed in the secret manager, not in the source code or Git repository. Keys are rotated every 90 days.

## Access control
Engineering access follows least privilege. All production role grants are reviewed quarterly. Emergency access expires after 4 hours and requires a ticket.

## Reporting incidents
Report potential security incidents to security@northstar.example within 30 minutes of discovery. The on-call security lead triages the incident, and critical incidents trigger an incident bridge.

## Data handling
Do not upload customer personal information to unapproved external AI providers. Redact sensitive fields before sharing diagnostic logs.
