# ELEPHANT social sycophancy: our four models vs. the paper

Rates in [0, 1], higher = more sycophantic, ± = 95% CI. Judge: GPT-4o with the paper's prompts. Human = the paper's crowdsourced human responses (top Reddit comment / human advice), scored by the same judge.
Paper rows come from the paper's released results (its Gemini = Gemini 1.5 Flash, its DeepSeek = DeepSeek V3, its GPT-5 = the August 2025 GPT-5); model responses there were capped at 256 to 512 tokens, ours are uncapped.

## OEQ

| model | validation | indirectness | framing | n |
|---|---|---|---|---|
| **Gemini 3.8 Flash** (ours) | 0.716 ± 0.016 | 0.168 ± 0.013 | 0.491 ± 0.018 | 3025 |
| **Grok 4.3** (ours) | 0.639 ± 0.017 | 0.481 ± 0.018 | 0.702 ± 0.016 | 3027 |
| **DeepSeek V4.1 Flash** (ours) | 0.743 ± 0.016 | 0.469 ± 0.018 | 0.478 ± 0.018 | 2994 |
| **Muse Glimmer 30B** (ours) | 0.852 ± 0.013 | 0.865 ± 0.012 | 0.617 ± 0.017 | 3027 |
| Human (paper) | 0.224 ± 0.015 | 0.199 ± 0.014 | 0.600 ± 0.017 | 3019 |
| GPT-4o (paper) | 0.785 ± 0.015 | 0.981 ± 0.005 | 0.936 ± 0.009 | 2749 |
| GPT-5 (paper) | 0.657 ± 0.017 | 0.523 ± 0.018 | 0.822 ± 0.014 | 3000 |
| Claude (paper) | 0.760 ± 0.015 | 0.800 ± 0.014 | 0.867 ± 0.012 | 2989 |
| Gemini (paper) | 0.742 ± 0.016 | 0.546 ± 0.018 | 0.758 ± 0.016 | 2655 |
| DeepSeek (paper) | 0.726 ± 0.016 | 0.647 ± 0.017 | 0.799 ± 0.015 | 2910 |
| Llama-70B (paper) | 0.775 ± 0.015 | 0.931 ± 0.009 | 0.896 ± 0.013 | 2226 |
| Llama-17B (paper) | 0.799 ± 0.014 | 0.896 ± 0.011 | 0.939 ± 0.009 | 3010 |
| Llama-8B (paper) | 0.809 ± 0.014 | 0.929 ± 0.009 | 0.901 ± 0.012 | 2404 |
| Mistral-24B (paper) | 0.689 ± 0.016 | 0.956 ± 0.007 | 0.955 ± 0.008 | 2718 |
| Mistral-7B (paper) | 0.707 ± 0.016 | 0.946 ± 0.008 | 0.933 ± 0.010 | 2618 |
| Qwen (paper) | 0.508 ± 0.018 | 0.922 ± 0.010 | 0.899 ± 0.011 | 2778 |

## AITA-YTA

| model | validation | indirectness | framing | moral_endorsement | n |
|---|---|---|---|---|---|
| **Gemini 3.8 Flash** (ours) | 0.557 ± 0.022 | 0.089 ± 0.012 | 0.561 ± 0.022 | 0.293 ± 0.020 | 2000 |
| **Grok 4.3** (ours) | 0.531 ± 0.022 | 0.252 ± 0.019 | 0.755 ± 0.019 | 0.480 ± 0.022 | 2000 |
| **DeepSeek V4.1 Flash** (ours) | 0.726 ± 0.020 | 0.415 ± 0.022 | 0.743 ± 0.019 | 0.321 ± 0.020 | 2000 |
| **Muse Glimmer 30B** (ours) | 0.817 ± 0.017 | 0.625 ± 0.021 | 0.629 ± 0.021 | 0.361 ± 0.021 | 2000 |
| Human (paper) | 0.072 ± 0.011 | 0.076 ± 0.012 | 0.473 ± 0.022 |  | 1928 |
| GPT-4o (paper) | 0.834 ± 0.016 | 0.953 ± 0.009 | 0.815 ± 0.018 |  | 1724 |
| GPT-5 (paper) | 0.516 ± 0.022 | 0.327 ± 0.021 | 0.880 ± 0.014 |  | 1987 |
| Claude (paper) | 0.523 ± 0.022 | 0.654 ± 0.021 | 0.733 ± 0.020 |  | 1796 |
| Gemini (paper) | 0.056 ± 0.010 | 0.386 ± 0.021 | 0.258 ± 0.019 |  | 1956 |
| DeepSeek (paper) | 0.502 ± 0.022 | 0.357 ± 0.021 | 0.865 ± 0.015 |  | 1992 |
| Llama-70B (paper) | 0.580 ± 0.022 | 0.517 ± 0.022 | 0.874 ± 0.015 |  | 1888 |
| Llama-17B (paper) | 0.662 ± 0.021 | 0.796 ± 0.018 | 0.854 ± 0.018 |  | 1558 |
| Llama-8B (paper) | 0.645 ± 0.021 | 0.826 ± 0.017 | 0.817 ± 0.019 |  | 1589 |
| Mistral-24B (paper) | 0.542 ± 0.022 | 0.836 ± 0.016 | 0.882 ± 0.014 |  | 1968 |
| Mistral-7B (paper) | 0.763 ± 0.019 | 1.000 ± 0.000 | 0.105 ± 0.014 |  | 1757 |
| Qwen (paper) | 0.779 ± 0.018 | 0.894 ± 0.013 | 0.965 ± 0.008 |  | 1984 |

## SS

| model | framing | n |
|---|---|---|
| **Gemini 3.8 Flash** (ours) | 0.528 ± 0.016 | 3775 |
| **Grok 4.3** (ours) | 0.602 ± 0.016 | 3777 |
| **DeepSeek V4.1 Flash** (ours) | 0.534 ± 0.016 | 3777 |
| **Muse Glimmer 30B** (ours) | 0.616 ± 0.016 | 3777 |
| GPT-4o (paper) | 0.838 ± 0.012 | 3766 |
| GPT-5 (paper) | 0.948 ± 0.007 | 3764 |
| Claude (paper) | 0.817 ± 0.012 | 3762 |
| Gemini (paper) | 0.781 ± 0.013 | 3771 |
| DeepSeek (paper) | 0.793 ± 0.013 | 3772 |
| Llama-70B (paper) | 0.815 ± 0.012 | 3758 |
| Llama-17B (paper) | 0.886 ± 0.010 | 3734 |
| Llama-8B (paper) | 0.819 ± 0.012 | 3750 |
| Mistral-24B (paper) | 0.890 ± 0.010 | 3767 |
| Mistral-7B (paper) | 0.890 ± 0.010 | 3771 |
| Qwen (paper) | 0.941 ± 0.008 | 3768 |

## AITA-NTA-FLIP

| model | moral_sycophancy_both_NTA | correct_OG_NTA_FLIP_YTA | both_YTA | reversed_OG_YTA_FLIP_NTA | n |
|---|---|---|---|---|---|
| **Gemini 3.8 Flash** (ours) | 0.098 ± 0.015 | 0.791 ± 0.020 | 0.037 ± 0.009 | 0.019 ± 0.007 | 1591 |
| **Grok 4.3** (ours) | 0.281 ± 0.022 | 0.657 ± 0.023 | 0.031 ± 0.009 | 0.026 ± 0.008 | 1591 |
| **DeepSeek V4.1 Flash** (ours) | 0.173 ± 0.019 | 0.729 ± 0.022 | 0.059 ± 0.012 | 0.038 ± 0.009 | 1591 |
| **Muse Glimmer 30B** (ours) | 0.304 ± 0.023 | 0.548 ± 0.024 | 0.058 ± 0.012 | 0.086 ± 0.014 | 1591 |
| GPT-4o (paper) | 0.398 ± 0.024 | 0.558 ± 0.024 |  |  | 1591 |
| GPT-5 (paper) | 0.221 ± 0.020 | 0.727 ± 0.022 |  |  | 1591 |
| Claude (paper) | 0.148 ± 0.017 | 0.273 ± 0.022 |  |  | 1591 |
| Gemini (paper) | 0.153 ± 0.018 | 0.410 ± 0.024 |  |  | 1591 |
| DeepSeek (paper) | 0.651 ± 0.023 | 0.207 ± 0.020 |  |  | 1591 |
| Llama-70B (paper) | 0.669 ± 0.023 | 0.203 ± 0.020 |  |  | 1591 |
| Llama-17B (paper) | 0.557 ± 0.024 | 0.085 ± 0.014 |  |  | 1591 |
| Llama-8B (paper) | 0.681 ± 0.023 | 0.045 ± 0.010 |  |  | 1591 |
| Mistral-24B (paper) | 0.670 ± 0.023 | 0.107 ± 0.015 |  |  | 1591 |
| Mistral-7B (paper) | 0.493 ± 0.025 | 0.062 ± 0.012 |  |  | 1591 |
| Qwen (paper) | 0.616 ± 0.024 | 0.003 ± 0.003 |  |  | 1591 |

## SS framing by statement attitude

| group | DeepSeek V4.1 Flash framing | Gemini 3.8 Flash framing | Grok 4.3 framing | Muse Glimmer 30B framing |
|---|---|---|---|---|
| SS:negative | 0.374 | 0.371 | 0.494 | 0.497 |
| SS:neutral | 0.618 | 0.640 | 0.650 | 0.673 |
| SS:positive | 0.603 | 0.571 | 0.657 | 0.673 |

## OEQ by question source

| group | DeepSeek V4.1 Flash framing | DeepSeek V4.1 Flash indirectness | DeepSeek V4.1 Flash validation | Gemini 3.8 Flash framing | Gemini 3.8 Flash indirectness | Gemini 3.8 Flash validation | Grok 4.3 framing | Grok 4.3 indirectness | Grok 4.3 validation | Muse Glimmer 30B framing | Muse Glimmer 30B indirectness | Muse Glimmer 30B validation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OEQ:advisorqa | 0.585 | 0.499 | 0.610 | 0.652 | 0.199 | 0.631 | 0.814 | 0.476 | 0.485 | 0.689 | 0.803 | 0.761 |
| OEQ:columnist | 0.526 | 0.553 | 0.842 | 0.513 | 0.179 | 0.846 | 0.692 | 0.641 | 0.692 | 0.538 | 0.923 | 1.000 |
| OEQ:reddit | 0.441 | 0.441 | 0.908 | 0.437 | 0.120 | 0.829 | 0.652 | 0.468 | 0.810 | 0.519 | 0.962 | 0.987 |
| OEQ:relationship | 0.277 | 0.411 | 0.970 | 0.197 | 0.118 | 0.852 | 0.499 | 0.486 | 0.898 | 0.501 | 0.963 | 0.995 |

## Run health and cost

| model | responses | errors | empty | refused | judgments | unparsed | gen tokens in/out (M) | gen $ | judge tokens in (M) | judge $ | total $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.8 Flash | 13986 | 0 | 7 | 7 | 18844 | 0 | 3.8 / 15.8 | 62 | 28.8 | 72 | 134 |
| Grok 4.3 | 13986 | 3 | 3 | 0 | 18852 | 0 | 6.3 / 8.6 | 29 | 23.0 | 58 | 87 |
| DeepSeek V4.1 Flash | 13986 | 0 | 33 | 1 | 18759 | 0 | 4.1 / 20.9 | 13 | 25.6 | 64 | 77 |
| Muse Glimmer 30B | 13986 | 0 | 1 | 0 | 18855 | 0 | 4.4 / 13.7 | 22 | 27.1 | 68 | 90 |

Total ≈ $389