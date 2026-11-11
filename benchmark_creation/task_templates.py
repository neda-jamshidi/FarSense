"""Prompt templates that turn a scenario and its generated tasks into the FarSense test items
(the `prompt` field of the dataset), kept verbatim.

Instructions are in English; all task content (context, question, options, ...) is in Farsi.
The model must answer with exactly one JSON object as shown in each template.
"""

PROMPTS: dict[str, str] = {

    # -------------------------------------------------------------
    # 1. Multiple‑Choice
    # -------------------------------------------------------------
    "multiple_choice": r"""Task — Multiple‑Choice (Commonsense, Farsi)

Based on the following Farsi Context, answer the question by selecting the correct option. Provide the answer in the JSON format below.

Context (Farsi):
{context_block_if_any}

Question (Farsi)
{question}

Options (Farsi):
{options}

Give the answer in the following JSON format:
{{"multiple_choice_question_answer_predict":"A" | "B" | "C" | "D"}}
""",

    # -------------------------------------------------------------
    # 2. True / False  #1
    # -------------------------------------------------------------
    "true_false_1": r"""Task — True/False (Commonsense, Farsi)

Based on the Context, state whether the statement is True or False. Provide the answer in the JSON format below.

Context (Farsi):
{context_block_if_any}

Statement (Farsi):
{statement}

Give the answer in the following JSON format:
{{"true_false_1_answer_predict": true | false}}
""",

    # -------------------------------------------------------------
    # 3. True / False  #2
    # -------------------------------------------------------------
    "true_false_2": r"""Task — True/False (Commonsense, Farsi)

Based on the Context, state whether the statement is True or False. Provide the answer in the JSON format below.

Context (Farsi):
{context_block_if_any}

Statement (Farsi):
{statement}

Give the answer in the following JSON format:
{{"true_false_1_answer_predict": true | false}}
""",

    # -------------------------------------------------------------
    # 4. Cloze Completion
    # -------------------------------------------------------------
    "cloze": r"""Task — Cloze Completion (Commonsense, Farsi)

Based on the Context, supply the Farsi phrase that best completes the blank «_____». Provide the answer in the JSON format below.

Context (Farsi):
{context_block_if_any}

Statement (Farsi):
{sentence_with_blank}

Give the answer in the following JSON format:
{{"cloze_answer_predict":"<missing Farsi phrase>"}}
""",

    # -------------------------------------------------------------
    # 5. Cause → Effect
    # -------------------------------------------------------------
    "cause_effect": r"""Task — Cause → Effect (Commonsense, Farsi)

Based on the Context and the given Cause, write the direct effect in Farsi. Provide the answer in the JSON format below.

Context (Farsi):
{context_block_if_any}

Cause (Farsi):
{cause_text}

Give the answer in the following JSON format:
{{"cause_effect_effect_predict":"<effect in Farsi>"}}
""",

    # -------------------------------------------------------------
    # 6. Counter‑factual
    # -------------------------------------------------------------
    "counterfactual": r"""Task — Counter‑factual Q&A (Commonsense, Farsi)

Answer the question in Farsi in ≤2 sentences (≤40 words), based on the Context. Provide the answer in the JSON format below.

Context (Farsi):
{context_block_if_any}

Counter‑factual (Farsi):
{counterfactual_question}

Give the answer in the following JSON format:
{{"counterfactual_answer_predict":"<Farsi answer, ≤40 words>"}}
""",

    # -------------------------------------------------------------
    # 7. Why Explanation
    # -------------------------------------------------------------
    "why": r"""Task — Why? Explanation (Commonsense, Farsi)
    
Based on the Context, answer the provided “why” question in Farsi in ≤2 sentences (≤40 words). Give the answer in the JSON format below.

Context (Farsi):
{context_block_if_any}

Question (Farsi):
{why_prompt}

Give the answer in the following JSON format:
{{"explanation_answer_predict":"<Farsi explanation>"}}
""",
}
