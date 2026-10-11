# API Rate Limits

## Overview

All API requests are subject to rate limits enforced per API key. Limits reset on a rolling
60-second window unless stated otherwise.

## Request limits

| Plan       | Requests per minute | Requests per day |
|------------|--------------------:|-----------------:|
| Free       |                  60 |            1,000 |
| Starter    |                 300 |           10,000 |
| Pro        |               1,000 |          100,000 |
| Enterprise |           Unlimited |        Unlimited |

## Token limits

Each request is also subject to a token budget:

- **Input tokens per request:** 8,192 (Free / Starter), 32,768 (Pro / Enterprise)
- **Output tokens per request:** 2,048 (Free / Starter), 8,192 (Pro / Enterprise)
- **Tokens per minute (TPM):** 40,000 (Free), 200,000 (Starter), 800,000 (Pro), unlimited (Enterprise)

## Concurrent request limits

| Plan       | Max concurrent requests |
|------------|------------------------:|
| Free       |                       2 |
| Starter    |                       5 |
| Pro        |                      20 |
| Enterprise |               Unlimited |

## Rate limit headers

Every API response includes the following headers:

```
X-RateLimit-Limit-Requests: 300
X-RateLimit-Remaining-Requests: 287
X-RateLimit-Reset-Requests: 2024-11-01T12:00:34Z
X-RateLimit-Limit-Tokens: 200000
X-RateLimit-Remaining-Tokens: 183412
X-RateLimit-Reset-Tokens: 2024-11-01T12:00:12Z
```

## Handling 429 errors

When a limit is exceeded the API returns HTTP `429 Too Many Requests` with a JSON body:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "You have exceeded your requests-per-minute limit.",
    "retry_after_seconds": 18
  }
}
```

Respect the `retry_after_seconds` value. Implement exponential back-off with jitter; do not
retry in a tight loop.

## Burst allowance

Pro and Enterprise plans permit a short burst of up to 2× the per-minute request limit for
at most 10 seconds before throttling begins. Free and Starter plans have no burst allowance.

## Limit increases

Enterprise customers may request higher limits by contacting support. Limit-increase requests
require a description of the workload, expected peak QPS, and average request size in tokens.
