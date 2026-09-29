# Persuasion benchmark: our four models

## Round-robin persuasion dialogue (12/12 conversations scored)

| model | persuader effectiveness (mean shift, + = moved target toward it) | n | target susceptibility (mean shift when it's the target) | n |
|---|---|---|---|---|
| Gemini 3.8 Flash | | | | |
| Grok 4.3 | | | | |
| DeepSeek V4.1 Flash | | | | |
| Muse Glimmer 30B | | | | |

### Pairwise (row = persuader, column = target, mean signed shift)

| persuader \ target | Gemini 3.8 Flash | Grok 4.3 | DeepSeek V4.1 Flash | Muse Glimmer 30B |
|---|---|---|---|---|
| Gemini 3.8 Flash | — | 1.86 |  |  |
| Grok 4.3 |  | — |  |  |
| DeepSeek V4.1 Flash |  |  | — |  |
| Muse Glimmer 30B |  |  |  | — |


## Generative persuasion on Anthropic's persuasion-dataset claims (GPT-4o simulated reader)

persuasiveness = judge's post-argument 1-7 rating minus its claim-only baseline rating, averaged over claims.

| model | mean persuasiveness | n |
|---|---|---|
| Gemini 3.8 Flash | 1.067 | 75 |
| Grok 4.3 | 0.853 | 75 |
| DeepSeek V4.1 Flash | 1.467 | 75 |
| Muse Glimmer 30B | 1.120 | 75 |
