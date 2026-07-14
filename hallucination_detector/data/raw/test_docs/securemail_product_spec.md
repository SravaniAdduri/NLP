# SecureMail Product Specification (v2.1)

## Overview
SecureMail is a web application for encrypted message delivery between registered users.
The service is designed for confidential internal communication.

## Core Features
1. Sender composes a message and selects one or more recipients.
2. Message content is encrypted before storage.
3. Recipient must sign in to the SecureMail portal to view messages.
4. Messages expire after a retention window configured by administrators.

## Authentication and Access
- Login supports email + password.
- Two-factor authentication (2FA) is optional in v2.1.
- Session timeout is 20 minutes of inactivity.
- Password reset links expire after 30 minutes.

## Security Details
- Data at rest uses AES-256 encryption.
- Data in transit uses TLS 1.3.
- Audit logs are retained for 180 days.
- Administrators can export audit logs in CSV format.

## Limitations
- No attachment support in v2.1.
- No native mobile app in v2.1.
- No cross-tenant message routing.

## Release Notes
- v2.1 release date: 2025-09-15
- Added: bulk recipient selection
- Fixed: delayed notification issue for large recipient groups
