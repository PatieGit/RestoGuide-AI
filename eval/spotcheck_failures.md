# Spot-check log

Retrieval misses noticed outside the benchmark. Per `IMPLEMENTATION.md`, a spot-check
failure is logged here and promoted into `benchmark.json` rather than triggering a
strategy change on its own — one unrepresentative query is not evidence.

## Phase 3 — answer-content spot-check: **passed**

`"what temperature should cold food be kept at?"` retrieved `rec-010` at rank 1
(score 0.8345) despite `4°C` appearing only in the answer body, which is the known
blind spot of question-only embedding. No failure to log. Promoted into the
benchmark anyway as `para-02`, since it probes the strategy's weakest point.

## Phase 5 — para-11: floor miscalibration, **fixed**

`"What should I do with the raw fish before I go home?"` retrieved the correct record
(`rec-008`, closing the kitchen) at rank 1, but scored **0.7884** — below the flat
0.80 floor in force at the time. Retrieval was right; the guardrail was wrong, and
staff would have been told the manual has nothing on it.

This was the case that decided the floor configuration. The 30-pair distribution
showed the weakest legitimate match (0.7884) sitting *below* the strongest
out-of-scope match (0.7984), so no flat threshold could separate them. Switching to
the relative floor gated at 0.75 answers this question and every other legitimate
one, while all six out-of-scope queries are still refused.

**No open spot-check failures.**
