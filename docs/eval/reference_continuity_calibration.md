# Reference continuity calibration set

Hand-coded page-to-page continuity examples from the reference books in
`reference/0-2/` and `reference/3-6/`, scored against the taxonomy in
`docs/superpowers/specs/2026-07-08-picture-book-spread-continuity-design.md`.
This is the "professional baseline" the `narrative_continuity_judge.py` eval
script is calibrated against — it is not exhaustive (only the pages already
captured as reference material were coded), and gaps in captured pages are
noted where they occur.

## Taxonomy used

| Category | Meaning |
| --- | --- |
| Recurring character/object | Same character/object reappears across the transition |
| Consequence-of-prior-action | This page's event follows causally from the previous page's |
| Setting persistence | Same location/event context carries across the transition |
| Emotional arc progression | A clear emotional beat change (e.g. worry → relief) |
| **None** | No detectable continuity — pages are self-contained (see note below) |

**Note — taxonomy refinement.** The original spec listed only the four
positive categories. Coding "From Head to Toe" surfaced the need for an
explicit **None** label: some books (educational call-and-response board
books) intentionally have zero cross-page continuity, and a judge forced to
pick a positive category for every transition would produce a false
signal. `narrative_continuity_judge.py` should allow `"none"` as a valid
classification, not just the four positive categories.

## Coded transitions

### 陶樂蒂的開學日 (age 3-6)

| Spread pair | Category | Notes |
| --- | --- | --- |
| 5-6 → 7-8 | Setting persistence | 5-6: principal announces "從今天開始,你們就是小學生了" to the crowd. 7-8: a new child, Stephen, cries "我不要上學!" — different character, but same first-day-of-school event and crowd. |
| 7-8 → 9-10 | Setting persistence | 7-8 ends on Stephen crying. 9-10 introduces James showing off his missing teeth — no causal link to Stephen, but same schoolyard, same background cast visible across both. |

Pattern: cohesion comes from an **ensemble vignette structure** (recurring
cast in one persistent setting), not a causal chain between individual
character beats.

### 正能量企鵝繪本夏日的朋友 (age 3-6)

| Spread pair | Category | Notes |
| --- | --- | --- |
| 5-6 → 7-8 | Recurring object; Consequence-of-prior-action | 5-6: something falls from the sky, penguin is curious. 7-8: "哦~這是種子...埋進土裡種種看吧" — directly resolves 5-6's mystery object and initiates the action. |
| 7-8 → 11-12 *(gap: 9-10 not captured)* | Recurring object; Consequence-of-prior-action; Emotional arc progression | 7-8 ends with the seed just planted. 11-12: "發芽了耶!太棒了!" — the same seed has sprouted; penguin is praised and "好開心好開心". |
| 11-12 → 13-14 | Recurring object; Emotional arc progression | 13-14 speculates the sprout will grow "比小企鵝更高、更壯" — same sprout, tone shifts from joy to anticipation. |
| 13-14 → 15-16 | Recurring object; Consequence-of-prior-action | 15-16 pays off 13-14's speculation literally: the sprout grows taller than the tit, then taller than the big penguin. |

Pattern: **the clearest positive example in the set** — a single recurring
object (the seed/sprout) chains all four captured transitions together
through consequence, with an emotional arc (curiosity → hope → pride →
wonder) riding on top of it.

### 我學會等待 (age 3-6)

| Spread pair | Category | Notes |
| --- | --- | --- |
| 5-6 → 9-10 *(gap: 7-8 not captured)* | Setting persistence; Recurring character | 5-6 establishes Chichi's morning routine. 9-10 introduces Mia the hen as a recurring companion with an established "老地方" (usual spot) ritual. |
| 9-10 → 13-14 | Consequence-of-prior-action; Emotional arc progression | 13-14 directly breaks the routine established in 9-10 — "但是,某天早上,米亞沒有出來迎接奇奇" — worry replaces the earlier warmth. |
| 13-14 → 17-18 | Consequence-of-prior-action; Emotional arc progression | 17-18 resolves the search from 13-14 ("原來妳在這裡!"), relief immediately complicated by Mia's strange stillness — sets up the next narrative beat. |

Pattern: the strongest **causal, single-thread** continuity in the 3-6 set
— each transition depends on specific prior-page information, not just a
shared setting.

### From Head to Toe (age 0-2, primary reliable source)

| Page pair | Category | Notes |
| --- | --- | --- |
| "I am a penguin... Can you do it?" → "I am a giraffe..." *(after its own "I can do it!" resolution)* | **None** | Each animal unit is fully self-contained (prompt → child imitates → "I can do it!"). No relationship to the animal before or after. |
| "I am a giraffe..." → "I am a buffalo..." | **None** | Same call-and-response pattern restarts with no reference to the giraffe. |

Pattern: a **repeated structural rhythm**, not narrative continuity —
important negative evidence that not all picture books rely on cross-page
continuity; it's an age/genre-dependent device, not a universal one.

### The Very Hungry Caterpillar (age 0-2, corroborating source)

| Page pair | Category | Notes |
| --- | --- | --- |
| "He ate one apple. But he was still hungry." → "He ate two pears. But he was still hungry." | Recurring character; Consequence-of-prior-action | Same subject ("He"), escalating count, repeated refrain literally carries the hunger state forward. |
| "He ate two pears..." → "He ate three plums..." | Recurring character; Consequence-of-prior-action | Same escalating-cumulative pattern. |
| "He ate five oranges..." → "He ate a lot of things." | Consequence-of-prior-action | The escalation turns into a binge — the turning point the eventual stomachache pays off (not captured in this sample). |

Pattern: continuity through **cumulative refrain and escalation**, a
different mechanism from 企鵝's object-based chain but the same underlying
principle — a recurring subject/phrase plus a change each page.

## Summary

| Book | Age band | Dominant continuity mechanism |
| --- | --- | --- |
| 正能量企鵝繪本夏日的朋友 | 3-6 | Recurring object + consequence chain |
| 我學會等待 | 3-6 | Single-thread causal continuity |
| 陶樂蒂的開學日 | 3-6 | Ensemble/setting persistence, no causal chain |
| The Very Hungry Caterpillar | 0-2 | Cumulative refrain/escalation |
| From Head to Toe | 0-2 | None — self-contained call-and-response |

This spread (5 books, 5 different mechanisms, including a genuine zero-continuity
control) is what `narrative_continuity_judge.py` should be validated
against before it's trusted to score pipeline output.
