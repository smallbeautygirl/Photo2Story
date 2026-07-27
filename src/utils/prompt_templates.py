VLM_DESCRIBE_PHOTO = (
    "Describe this photo in 2-3 sentences. "
    "Focus on: people present, their actions, the location, and the mood. "
    "Be specific and concrete."
)

LLM_SCORE_RELEVANCE = """\
You are helping select photos for a storybook about: "{context}"

Photo description: "{description}"

Rate how relevant this photo is to the storybook context on a scale from 0.0 to 1.0.
Reply with ONLY a decimal number between 0.0 and 1.0, nothing else."""

LLM_INFER_THEME = """\
Below are short descriptions of {k} photos that share a moment, trip, or theme.

{descriptions}

In one short phrase (5-10 words), name the unifying activity, event, or theme these photos depict. Be specific and concrete.

Reply with ONLY the phrase, no quotes, no trailing punctuation.

Example replies:
a child's first swimming lesson
family visit to the apple orchard
birthday party at the park"""

LLM_INFER_ORDER = """\
You have {k} photos from a personal trip. Their descriptions (in no particular order) are:

{descriptions}

The trip context: "{context}"

List the numbers 1 to {k} in the order these photos most likely occurred during the trip.
Reply with ONLY the numbers separated by commas, e.g.: 2,1,3,4"""

LLM_CAUSAL_INFERENCE = """\
The following {k} photo descriptions are in story order from a trip about: "{context}"

{descriptions}

In 2-3 sentences, describe the narrative arc: what happened first, what was the turning point, and how it ended. Focus on cause-and-effect relationships between scenes."""

# Reading-level presets control how short and simple the generated story is.
# "simple" targets a real picture-book register (few words, common vocabulary);
# "standard" preserves the original 2-3 sentence behaviour for ablation baselines.
READING_LEVELS = {
    "simple": {
        "age": "ages 3 to 5",
        "sentences": "exactly one short sentence",
        "length_hint": "at most ~24 characters for Chinese/Japanese, or ~12 words for English",
        "vocab": (
            "Use only the most common everyday words a preschooler already knows. "
            "No idioms, no rare or abstract words, no subordinate clauses."
        ),
    },
    "standard": {
        "age": "ages 6 to 8",
        "sentences": "two to three short sentences",
        "length_hint": "at most ~45 characters for Chinese/Japanese, or ~35 words for English",
        "vocab": "Use simple, warm, child-friendly vocabulary.",
    },
}

LLM_STORY_GENERATION = """\
You are a children's storybook author writing for {age}. Write a {k}-page storybook in {style} style.
Write the entire story in the language identified by BCP 47 locale code "{language}" (e.g. "en" = English, "zh-tw" = Traditional Chinese, "zh-cn" = Simplified Chinese, "ja" = Japanese).

Trip context: "{context}"
Narrative arc: "{narrative}"

Photo descriptions (one per page):
{descriptions}

Rules:
- Write exactly {k} pages
- Each page: {sentences} ({length_hint})
- {vocab}
- Every sentence must be grammatically complete and natural in the target language -- do not drop prepositions, particles, or verb-complement markers just to shorten a sentence (e.g. Chinese needs 在/著/地 where a natural sentence would use them, not a telegraphic string of nouns)
- Pages must connect naturally: use techniques like a recurring character or object reappearing, a consequence following from the previous page's event, or the same setting carrying across pages -- vary which technique you use rather than repeating one every page
- Warm, child-friendly tone
- All page text must be written in language "{language}" (BCP 47)
- Return ONLY a JSON array of strings, one string per page

Example format:
["Page 1 text here.", "Page 2 text here.", "Page 3 text here."]"""

# Style first: CLIP truncates at 77 tokens, so the style fragment (and its trigger
# token) must lead the prompt or it gets dropped and the look reverts to generic.
SD_PROMPT_TEMPLATE = (
    "{style_prompt}, children's storybook illustration. "
    "{scene} warm colors, detailed"
)
