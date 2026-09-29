# Research design

## Question

How robust is multi-agent LLM deliberation when a subset of agents is secretly adversarial? We test whether honest-agent defection is better predicted by the absolute number of deceivers, their proportion in the group, or an interaction with total group size.

## Protocol

For each Humanity's Last Exam question, all agents first answer independently in round 0 and then write a private reflection after viewing the public board. In later rounds, agents respond synchronously using their previous private reflection and the preceding public board, then update their reflection after all current-round responses are posted. Finally, each agent casts a private vote without seeing the other votes.

Deceivers follow the same communication protocol, but privately receive the reference answer and are instructed to persuade the group toward an incorrect answer without revealing their role. They may change arguments or target answers as the discussion evolves. In the coordination condition, deceivers additionally receive a private channel for planning before and during deliberation.

The bounded-memory design prevents an ever-growing public transcript while allowing each agent to retain information it considers important. See [`sabotage/PROTOCOL.md`](sabotage/PROTOCOL.md) for the rendered prompts at every stage.

## Primary outcome

Honest defection is the share of honest agents that answer correctly in round 0 but cast an incorrect final vote. We additionally measure group accuracy, recovery from incorrect intermediate answers, the timing of first defections, persuasion tactics, and whether honest agents detect coordination.

## Experimental factors

- Homogeneous model family: Gemini 3.8 Flash, Grok 4.3, DeepSeek V4.1 Flash, or Muse Glimmer 30B.
- Group composition: 12 honest/deceiver configurations spanning 2 to 21 total agents.
- Adversarial proportion: 0, 1/5, 1/3, or 3/7.
- Deceiver communication: independent or privately coordinated.
- Heterogeneous pairing: Gemini or Muse Glimmer honest agents crossed with DeepSeek or Grok deceivers.

The committed question-selection files stratify HLE questions by the number of correct responses in four independent attempts. GPT-5.4 grades answers using the official HLE instructions; it is not a deliberating-agent condition.
