# API Guide

## 1. Introduction

This guide describes the REST API for the inference service. It covers authentication,
request construction, response formats, pagination, versioning, and error handling.

## 2. Base URL

All API endpoints are served over HTTPS from the base URL:

```
https://api.example.com/v1
```

Requests made over plain HTTP are not supported. Always use HTTPS to ensure data is
encrypted in transit.

## 3. Authentication

Include your API key as a bearer token in every request:

```
Authorization: Bearer <your-api-key>
```

API keys are generated in the dashboard under **Settings → API Keys**. Keys are scoped to
a single account and can be revoked at any time. Never embed a key in client-side code or
commit it to version control.

## 4. Request Format

Send requests as JSON with `Content-Type: application/json`. The top-level fields are:

| Field         | Required | Type    | Description                                          |
|---------------|----------|---------|------------------------------------------------------|
| `model`       | Yes      | string  | Model identifier (e.g. `balanced-v2`).               |
| `messages`    | Yes      | array   | Array of `{ "role": "user" | "assistant", "content": "..." }` pairs. |
| `max_tokens`  | No       | integer | Upper bound on completion length. Default: 1024.     |
| `temperature` | No       | float   | Sampling temperature, 0.0–2.0. Default: 1.0.         |
| `stream`      | No       | boolean | Set `true` to receive a server-sent-event stream.    |

### Example request

```json
{
  "model": "balanced-v2",
  "messages": [
    { "role": "user", "content": "What are the rate limits for the API?" }
  ],
  "max_tokens": 512,
  "temperature": 0.7
}
```

## 5. Response Format

Successful responses return HTTP `200 OK` with a JSON body:

```json
{
  "id": "cmpl-9kXw2mNpQr",
  "model": "balanced-v2",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 142,
    "total_tokens": 160
  }
}
```

## 6. Rate Limits

Requests are rate-limited per API key. The limit tier is determined by your plan:

| Plan       | Requests per minute | Requests per day |
|------------|--------------------:|-----------------:|
| Free       |                  60 |            1,000 |
| Starter    |                 300 |           10,000 |
| Pro        |               1,000 |          100,000 |
| Enterprise |           Unlimited |        Unlimited |

Exceeding a limit returns HTTP `429 Too Many Requests`. Inspect the `X-RateLimit-*` response
headers to track remaining quota and the reset timestamp. See `limits.md` for full details
including token-per-minute limits, concurrent request limits, and burst allowances.

## 7. Errors

Non-2xx responses include a JSON error object:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "You have exceeded your requests-per-minute limit.",
    "retry_after_seconds": 18
  }
}
```

Common error codes:

| Code                      | HTTP status | Description                                          |
|---------------------------|-------------|------------------------------------------------------|
| `invalid_api_key`         | 401         | Missing, malformed, or revoked API key.              |
| `rate_limit_exceeded`     | 429         | Per-minute or per-day request limit hit.             |
| `context_length_exceeded` | 400         | Prompt exceeds the model's maximum context window.   |
| `model_not_found`         | 404         | Requested model ID does not exist.                   |
| `server_error`            | 503         | Transient backend failure — retry with back-off.     |

## 8. Pagination

List endpoints return a paginated response. Use the `after` cursor parameter to fetch the
next page:

```
GET /v1/completions?limit=20&after=cmpl-9kXw2mNpQr
```

The response includes `has_more: true` and a `next_cursor` field when additional pages exist.

## 9. Versioning

The API version is embedded in the URL path (`/v1/`). Breaking changes increment the version
number. Deprecated endpoints include a `Sunset` header with the planned removal date.
Subscribe to the changelog at `docs.example.com/changelog` for advance notice of changes.
