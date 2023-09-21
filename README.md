# SentimentGPT


## Getting started 

Make sure you add your OPENAI_API_KEY to your environment variables before running docker-compose commands

```bash
docker-compose build && docker-compose up
``` 

## Swagger docs 

http://localhost:8000/docs

Sample curl command 

```curl
curl -X 'POST' \
  'http://0.0.0.0:8000/api/v1/sentiment/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  {
    "message": "The first human patient will soon receive a Neuralink device. This ultimately has the potential to restore full body movement. In the long term, Neuralink hopes to play a role in AI risk civilizational risk reduction by improving human to AI (and human to human) bandwidth by several orders of magnitude. Imagine if Stephen Hawking had had this.",
    "request_id": "fbb9f6e1-5b23-4f2d-a9c1-05932c405901"
  },
  {
    "message": "Chair Gensler: The crypto space is full of hucksters and non-compliance",
    "request_id": "218e8f60-f680-49fc-9d51-104f8ed8ec63"
  }
]'
```

