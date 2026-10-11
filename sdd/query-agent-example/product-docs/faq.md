# Frequently Asked Questions

## General

### What is this API?

This API provides programmatic access to our language model inference service. It accepts
natural-language prompts and returns completions, structured outputs, or embeddings depending
on the endpoint called.

### Which authentication method does the API use?

All requests must include a bearer token in the `Authorization` header:

```
Authorization: Bearer <your-api-key>
```

API keys are generated in the dashboard under **Settings → API Keys**. Keys do not expire
automatically, but you can revoke them at any time.

### Is there a free tier?

Yes. Free-tier accounts receive 1,000 requests per day and up to 60 requests per minute at
no cost. Free-tier keys are rate-limited and do not support burst allowances or concurrent
request counts above 2.

---

## Models

### Which models are available?

| Model ID              | Context window | Best for                          |
|-----------------------|---------------:|-----------------------------------|
| `fast-v1`             |         8,192  | Low-latency, short completions    |
| `balanced-v2`         |        32,768  | General-purpose tasks             |
| `reasoning-v1`        |        32,768  | Multi-step reasoning, coding      |
| `embedding-small-v1`  |           N/A  | Semantic search, classification   |

### How do I select a model?

Pass the `model` field in your request body:

```json
{
  "model": "balanced-v2",
  "messages": [{ "role": "user", "content": "Hello" }]
}
```

### Are model outputs deterministic?

At `temperature=0` outputs are highly consistent but not guaranteed to be identical across
API versions or infrastructure changes. For reproducible experiments, pin the model version
and store outputs explicitly.

---

## Billing

### How is usage calculated?

Usage is measured in tokens. One token is approximately four characters of English text.
Both prompt tokens (input) and completion tokens (output) count toward your usage. Embedding
requests count only input tokens.

### Where can I see my current usage?

Navigate to **Dashboard → Usage** to view per-day and per-month token consumption, broken
down by model and endpoint. Usage data is updated with a delay of up to five minutes.

### What happens if I exceed my plan limit?

- **Free / Starter:** Requests beyond the daily limit return `429 Too Many Requests` until
  the limit resets at midnight UTC.
- **Pro:** Overage requests are billed at the standard overage rate listed on the pricing page.
- **Enterprise:** Limits are negotiated contractually; contact your account manager.

---

## Errors

### What does error code `context_length_exceeded` mean?

Your prompt plus the requested completion length exceeds the model's maximum context window.
Reduce the length of the input, lower `max_tokens`, or switch to a model with a larger context.

### What does error code `invalid_api_key` mean?

The API key in the `Authorization` header is missing, malformed, or has been revoked. Check
that you are passing the key correctly and that it has not been deleted from the dashboard.

### I am receiving intermittent `503` errors. What should I do?

Intermittent `503 Service Unavailable` responses indicate a transient backend issue. Retry
the request with exponential back-off. If the errors persist for more than five minutes,
check the status page at **status.example.com**.

---

## Security

### Is data sent to the API stored or used for training?

Data submitted via the API is not used to train or fine-tune any model by default. Enterprise
Data Protection (EDP) agreements are available for customers who require contractual guarantees.
Review the privacy policy for full details.

### Does the API support end-to-end encryption?

All API traffic is encrypted in transit using TLS 1.2 or higher. Data at rest is encrypted
with AES-256. There is no option for client-managed encryption keys in the current release.
