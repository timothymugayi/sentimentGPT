PROMPT = """
Given an json markdown array of object statements 

What is the sentiment of each of the statements?
Ranking	On a scale of 1 to 10, how positive or negative is this review?
Opinion	What is your opinion of this statement?
Profile	What is the profile of the person who would write this review?

statements: 

```json
{statements}
```

Output response should be a Markdown code snippet formatted in the following schema json array 
consisting of the sentiment analysis of each of the given statements only

example schema:

```json
[{{
    "request_id": "Original request ID sent as part of the statement",
    "sentiment": "Neutral", 
    "ranking": 5,
    "opinion": "The statement presents an interesting possibility that should be further investigated.",
    "profile": "Someone interested in global events and current affairs."
}}, 
{{
    "request_id": "Original request ID sent as part of the statement",
    "sentiment": "Negative", 
    "ranking": 3,
    "opinion": "The statement expresses a negative opinion of the crypto space.",
    "profile": "Someone knowledgeable about the crypto space."
}}]
```
"""
