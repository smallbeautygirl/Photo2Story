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

LLM_STORY_GENERATION = """\
You are a children's storybook author. Write a {k}-page storybook in {style} style.

Trip context: "{context}"
Narrative arc: "{narrative}"

Photo descriptions (one per page):
{descriptions}

Rules:
- Write exactly {k} pages
- Each page: 2-3 sentences
- Pages must connect naturally (reference what happened before)
- Warm, child-friendly tone
- Return ONLY a JSON array of strings, one string per page

Example format:
["Page 1 text here.", "Page 2 text here.", "Page 3 text here."]"""

SD_PROMPT_TEMPLATE = (
    "{page_text} "
    "Children's storybook illustration, {style} art style, "
    "warm colors, detailed, high quality"
)
