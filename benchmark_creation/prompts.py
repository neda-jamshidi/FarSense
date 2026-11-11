"""Generation prompts used to build FarSense (Section 3.2 of the paper), kept verbatim.

prompt_1  passage -> everyday scenario: rewrites a Farsi Wikipedia paragraph into a
          100-300 word narrative that implies its facts.
prompt_2  scenario -> tasks: derives the six commonsense task types (JSON) from that narrative.

`{{SOURCE_TEXT}}` is replaced with the passage (prompt_1) or the scenario (prompt_2).
"""

SYSTEM_PROMPT = (
    "You are a meticulous Farsi-language dataset artisan who converts any "
    "given Persian passage into a structured, four-stage commonsense-reasoning "
    "dataset entirely in Farsi."
)

prompt_1 = """ 

You will receive one short Persian passage.

1. Extract every explicit factual statement from the passage (keep them concise and literal—no interpretation).
2. Convert each fact into a plausible, everyday commonsense scenario that implies the fact without stating it directly.
3. Using every fact from step 1, expand the scenario from step 2 into a 100- to 300-word Farsi narrative, adding natural detail or a minor hypothetical twist. Be sure every extracted fact is clearly included.

Return only the narrative from step 3—no labels, explanations, JSON, or extra text of any kind.

Text:  
{{SOURCE_TEXT}}

"""

prompt_2 = """ 
For the following narrative, create all six commonsense reasoning task types below, entirely in Farsi, each with its correct answer:
Make sure the question requires implicit understanding and is not answerable through rote memorization.

1) Multiple-choice QA 

   – Write one commonsense question with four choices (A–D)  
   – The question should require reasoning beyond direct sentence recall.  
   – Provide the correct option letter.  


2) True/False statements  
    Write two declarative statements:
   – One must be clearly true, based on commonsense reasoning.
   – One must be clearly false, based on contradiction or implausibility.
   – Each item: {"statement": "...", "answer": true|false}.  


3) Cause-and-Effect pairs

   - Write one plausible cause-effect pair based on the story.
   - Focus on why something happened or what its result was.
   - Use commonsense logic—not just temporal order.
   – One pair: {"cause": "...", "effect": "..."}.  


4) Cloze completion  
   – Choose a meaningful sentence and blank out a key word or phrase using «_____».  
   - The answer must be non-obvious without understanding the context.
   – Provide the missing text as "answer".  

5) Explanation generation  
   – Ask a “Why” or “How” question about a behavior, intention, or event.
   - Answer it in 1–2 short Farsi sentences.
   - The answer should show inference, not surface repetition.



6) Counterfactual reasoning  
   – Ask a “What if…” question that changes one detail in the scenario.
   - Predict a reasonable alternate outcome.


Quality gates apply to every item: the correct answer must never appear verbatim in the narrative (the “no-span” rule); each item must include a key named "cs_skill" whose value is one of physical, social, temporal, intentional, causal, or deontic to state which commonsense capacity is being exercised; each item must also contain a single-sentence Persian explanation under "why_cs" that clarifies what piece of world knowledge is required. If any item fails one of these three requirements, delete it and regenerate until it conforms. A weak example would be asking the hero’s age when it is explicitly given in the text; a strong example would ask why the hero chose the more distant bakery, which requires understanding a social norm about quality preferences.   

────────────────────────────────────────────
★ OUTPUT FORMAT – RETURN EXACTLY THIS JSON ★
--------------------------------------------
{

    "multiple_choice": {
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "B,
      "cs_skill": "social",
     "why_cs": "..."
    },
    "true_false": [
      {"statement": "...", "answer": true, "cs_skill": "social",  "why_cs": "..."},
      {"statement": "...", "answer": false, "cs_skill": "social",  "why_cs": "..."}
    ],
    "cause_effect": 
      {"cause": "...", "effect": "...", "cs_skill": "social",  "why_cs": "..."}
    ,
    "cloze": 
      {"sentence": "... «_____» ...", "answer": "...", "cs_skill": "social",  "why_cs": "..."}
    ,
    "explanation": {
      "prompt": "...",
      "answer": "...",
      "cs_skill": "social", 
      "why_cs": "..."
    },
    "counterfactual": {
      "question": "...?",
      "answer": "...",
      "cs_skill": "social", 
      "why_cs": "..."
    }

}

Do not output anything outside this JSON object.  
The JSON must be valid and parseable.
  
Text:  
{{SOURCE_TEXT}}

"""
