# Speaking Is an Action: Turning TTS into a Tool for Interactive Models

| Model | Right about what was heard, both conditions (history replay, /32) | First prompts that call the speech tool (/80) | First audio on three-sentence requests |
|---|:---:|:---:|:---:|
| Inkling 2 Mini | 24 → **32** | 8 → **80** | 1,866 ms → **540 ms** (direct text: 1,263 ms) |
| Inkling 2 Small | 27 → **32** | 72 → **80** | 2,403 ms → **535 ms** (direct text: 1,579 ms) |
| Qwen3.8-27B | 12 → **32** | 1 → **80** | 3,124 ms → **625 ms** (direct text: 2,637 ms) |

Every model has thinking on at the same level (low). The arrows go from a trimmed reply with a system note to `tts_start` with `heard_prefix`, from an optional call to `tool_choice: required`, and from `tts_start` through SGLang's parser to a streamed argument with a prefilled opening, which generates no thinking.

## TL;DR

Models keep getting smarter, yet people still talk to them one turn at a time. Human communication is interactive: people keep perceiving while they respond, cut in, and change course when cut off. SGLang Omni builds interactive systems from open models, starting with a cascade of ASR, LLM and TTS coordinated by a Session that continuously runs interaction management. The old cascade read aloud every token the LLM wrote, which left a bad choice: with thinking off the model is dumb, and with thinking on it is far too verbose to say out loud. We wanted the cascade to use tools, and after thinking it through, the best form was to make TTS itself the first one: to the LLM, speaking is a tool call no different from a Google search or a weather lookup. Now the LLM passes `tts_start` only the words worth saying. The Session decides when speech stops and tells the model, through `heard_prefix`, how much the user heard. Across Inkling 2 Mini, Inkling 2 Small and Qwen3.8-27B, streaming the tool argument and prefilling the call opening cut median first audio on three-sentence requests from 2,464 ms to 567 ms on average, against 1,826 ms for the old direct-text path. The first and last numbers are with thinking on at low; the prefilled opening skips the thinking.

## Why Interactive Models

Models keep getting smarter, but the way people work with them has barely changed: write a prompt, wait for a reply. People don't communicate that way. They share the same moment and setting and respond while they perceive: they answer before the other side has finished, rephrase when they see a frown, and change course the moment something changes. In conversation, gaps between turns are around 200 ms, shorter than the 600 ms it takes to plan a sentence (Levinson & Torreira, 2015), and about 40% of turn changes overlap (Heldner & Edlund, 2010). Turn-based systems flatten this into a walkie-talkie exchange and make the human adapt to the machine: say it in one breath, don't pause, don't interrupt, wait. They guess from silence when your input has ended, so they either cut you off when you pause to think or make every turn wait.

The deeper problem is the model itself. A language model experiences the world in a single thread: while the user talks, it waits; while it generates, the world stands still. It sees a prompt, not a stream. It can't notice that you hesitated or looked away, or that its answer is drifting, and it can't feel ten seconds pass. One view holds that a person is a thinking machine taking in the world every 200 ms. Thinking Machines Lab's interaction model keeps reading audio, video and text and producing its response in 200 ms steps, which is what makes it possible to learn when to respond, when to wait and when to stop.

Nor is interaction a layer that can be added afterwards. Every moment holds a decision: speak, wait, stop, or go look something up. These decisions take as much intelligence as the answer itself; hand them to silence thresholds and hand-written rules, and they will be dumber than the model and fall further behind as models improve. So the goal is not a faster turn-based system. Full duplex, perceiving and responding at once, is only the means; the real goal is a model that interacts, whose interaction grows with its intelligence across every modality it can perceive and produce.

## Our Cascade

Native full-duplex models, such as Thinking Machines Lab's interaction model, OpenAI's GPT-Live and Moshi (Défossez et al., 2024), learn interaction end to end, but they need strong omni base models, and open source has few. Turn-based omni models still respond one turn at a time. A cascade of VAD, ASR, LLM and TTS is built from parts open source has in abundance: each upgrades on its own as better models ship, and each failure can be traced to its part. The cost is lost signal: ASR drops tone of voice and merges speakers. SGLang Omni's [streaming cascade](https://www.sglang.io/blog/omni-interactive-runtime) coordinates the parts with a Session that continuously runs interaction management: it decides when the user has finished or cut in, stops playback, and keeps the history in line with what was actually played. The cascade is a bridge to an interactive model. Until open models can make every interaction decision on their own, our rule is: give the model the decisions it makes well, let the system coordinate the signals the model cannot yet perceive, and hand those back as models improve.

## Why TTS Should Be a Tool

Imagine you are driving an electric car and ask its in-car assistant what the weather is like today. What do you expect to hear, and what do you expect to see? You want to hear one sentence, "72°F today", and glance at the touchscreen for the rest of the week. Behind that sentence, the model has already looked the weather up. What you want to hear is short; what you want to see can be rich. This kind of interaction takes real intelligence: the model knows what to think about, what to show and what to say aloud.

Our cascade did not work this way. ASR, the LLM and TTS were nodes at the same level, and TTS read every token the LLM wrote. The surface problem was tokens TTS can't read: emoji, Markdown asterisks, tables. We filtered those out. The real problem is that open LLMs are trained for coding and talk like it: long, structured, thousands of tokens at a time. Turn thinking off and the model gets dumb: its answers go wrong and it handles tools poorly. Turn thinking on and it gets smart, but nobody wants to hear everything it writes.

Reading every token also hides what the user actually heard. When the user cut in, the old cascade trimmed the reply in the history to the played part, ended it with "…" and added a system note. The model then picks up this broken output in context: even when nothing interrupts it, it no longer gives a complete answer and ends with an ellipsis itself. Given 20-turn histories with every reply cut, the setup from [#3255](https://github.com/sgl-project/sglang-omni/pull/3255), Qwen3.8-27B with thinking off, as the old cascade ran it, ended 12.5% of its final answers in an ellipsis (16 of 128). With thinking on, none of the three models did ([Evaluation](#evaluation)).

So TTS can't read everything, and the model has to choose. We also wanted the cascade to call tools: the in-car assistant has to check the weather before it can answer. After thinking it through, we found TTS as a tool to be the best form for both, and TTS became our first tool. We demoted it from a pipeline node to a tool; to the LLM, `tts_start` is no different from a Google search or a weather lookup. The model decides what to say. The system decides when speech must stop and records what the user likely heard. A tool call is the interface between them, and this is our answer, for speaking, to the question the runtime post left open: how much interaction behavior the model should learn, and how much the system should coordinate.

## System Design

The Session, the LLM and the TTS work together under three rules:

- **Only the Session changes state.** The LLM issues actions and the TTS executes them, but only the Session decides whether speech is on. Nodes never message each other, so their views of the state can't drift.
- **Each action has one owner.** `tts_start` belongs to the LLM and `tts_close` to the Session, so nobody races over who stopped what.
- **Each call gets one answer.** One `tts_start` gets one tool response, so `tool_call_id` is the only ID we need.

| Action | Called by | What it does |
|---|---|---|
| `tts_start(text)` | The LLM | Splits the text into sentences, then synthesizes and plays them one by one |
| `tts_close` | The Session, when the user starts talking | Stops playback at once and clears the buffers. It is not in the LLM's tool list. |

A Session plays one speech at a time. A second `tts_start` during playback gets `BUSY` instead of replacing it.

<div align="center">
  <img src="images/tts-tool-sequence.png" alt="A tts_start call: the LLM calls tts_start and the Session has the TTS speak; the speech either plays out and is recorded as completed, or the user starts speaking and the Session sends tts_close to the TTS and stop to the LLM, recording interrupted with heard_prefix" width="90%">
</div>

Every `tts_start` ends in exactly one outcome:

<div align="center">
  <img src="images/tts-tool-outcomes.png" alt="Outcomes of tts_start: a refused call or one with no words to speak returns failed; an accepted speech ends as completed, interrupted with heard_prefix, or failed with heard_prefix" width="90%">
</div>

`heard_prefix` is the part of the text the playback record says the user heard ([how it is computed](#computing-heard_prefix)).

**Interruption takes a fast path.** When voice activity detection (VAD) detects the user speaking, the Session closes the TTS and stops the LLM at once, without asking the LLM. Routing the user's words through ASR and the LLM first would let more audio play, and that audio would be recorded as heard although the user, who was talking, missed it.

## Design Trade-offs

### Thinking On or Off?

This trade-off is where the design starts. Without thinking, the models generalize poorly to a tool like this (unfortunately, we found that with thinking off, models have almost no zero-shot ability to operate tools): Qwen3.8-27B, the one model here that can turn thinking off, called an optional `tts_start` in 1 of 80 first prompts, and in 15 of 80 when the system prompt added that the user hears only what the tool says. With thinking on, that one sentence was enough: the three models called the tool in 74 to 80 of 80 prompts, at low and at medium. Left to work it out from the tool's description alone, only Inkling 2 Small called it (72 and 73 of 80); Inkling 2 Mini and Qwen3.8-27B mostly answered in plain text. Because TTS is a tool, the thinking is never spoken. The cost is latency: at medium thinking, writing the whole text took a median of 1.5 s (Inkling 2 Mini) to 3.5 s (Qwen3.8-27B) before any audio. The shipped profiles therefore force the call and prefill its opening, which skips the thinking ([next section](#making-real-models-use-it)); Qwen3.8-27B's profile also turns thinking off, and Inkling, which cannot turn it off, runs at `reasoning_effort` 0.1. Making thinking cheap enough to keep on is still open.

| Measure (thinking low / medium) | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| Optional `tts_start` called, shipped system prompt (/80) | 8 / 8 | 72 / 73 | 1 / 3 |
| Optional `tts_start` called, one more sentence saying the user hears only the tool (/80) | 78 / 80 | 80 / 80 | 80 / 74 |
| Median time to first audio at medium, one sentence, SGLang's parser | 1.5 s | 2.0 s | 3.5 s |

### When Does a Call Get Its Answer?

A speech lasts seconds, and the model needs to know how it ended. Answering `tts_start` at once with "accepted" and reporting completion later gives every call two answers, and tying them together needs a second ID that every later operation must carry.

We answer only when the model needs to act on the answer: a refused call at once, an accepted call once, when the speech ends. One call, one result, no second ID. The model learns nothing while it speaks, which costs little because nothing prompts it during that time anyway.

### Who May Stop a Speech?

We considered an LLM-side close, a `tts_edit` that rewrites sentences in flight, letting a new `tts_start` replace the current one, and queueing concurrent starts. We chose the narrowest option: only VAD triggers a close, through the Session, and a second start gets `BUSY`.

The LLM doesn't know how far its speech has got and was never trained on an interface like this, so it can't be trusted to stop at the right moment. An edit needs sentence splitting, lock points, version numbers and a way to drop stale audio. A close already returns the stop position, so close-then-restart covers the same need with far less machinery. The cost: the model can't correct itself mid-speech.

### How Does the Model Learn What Was Heard?

The Session writes the stop position into the tool result and updates it as playback reports arrive, using sentence boundaries by default and word timings when an aligner is on. The next prompt carries the new input and that result. An interrupted speech can enter the history in three ways:

| Approach | What the model sees |
|---|---|
| Trim the reply to what was heard (old cascade) | An assistant reply ending in an ellipsis, plus a system note |
| Put an interruption note in the user message | The heard text, inside the user's account of the interruption |
| Keep the call, put `heard_prefix` in the tool response (ours) | The generated text in the call, the heard text in its result |

Ours keeps what the model wrote apart from what was played; the call holds everything the model had written when it was cut. In one replay with Qwen3.8-27B, the call said "The parcel code is V2P2." and `heard_prefix` ended at "The pickup desk is next to the main entrance." Asked whether it had read the code aloud, the model's next `tts_start` said "Not yet read. The code is V2P2." In the same replay Inkling 2 Mini said "Not yet read; the code is V2P2." and Inkling 2 Small said "Not yet read. Your code is V2P2." (all three with thinking at low).

## Making Real Models Use It

**Forcing the call.** With `tool_choice: required`, every model called `tts_start` in 80 of 80 first prompts, at both thinking levels. If a server ignores `tool_choice` and the model writes plain text, that text is spoken as if it were a `tts_start` call and reported the same way.

**Ending a turn.** If every prompt requires a call, the model can't choose silence. We tried a `silent` tool. On a Session's first input the models used it well: they chose silence for 12 of 480 plain questions (all 12 from Qwen3.8-27B) and for 454 of 480 inputs that asked for no reply. After a completed speech they did not: prompted again with nothing new to answer, they spoke again in 130 of 480 samples, and in 130 of 480 with a `silent` action instead, about four times in five with the same answer. Inkling 2 Small spoke again in about two thirds of its samples with the `silent` tool. So the system ends the turn after one successful speech per user input and puts the result in the next prompt. The model is prompted again only if its call was refused or the speech failed, at most four times per input.

**First-audio latency.** SGLang's stock tool-call parser (`qwen3_coder` for Qwen3.8-27B, `inkling2` for Inkling 2 Mini and Inkling 2 Small) releases the text argument only at its closing tag, so synthesis waits for the whole argument. Two changes fix that:

1. Our chat service streams the `text` argument as the model writes it, so the first sentence is synthesized while the rest is still being written.
2. We [prefill the call opening](https://github.com/sgl-project/sglang-omni/pull/4472) as an assistant message for the model to continue (`continue_final_message: true`, `tool_choice: auto`), so decoding starts at the spoken text. For Qwen3.8-27B the opening is the 14 tokens `<tool_call>\n<function=tts_start>\n<parameter=text>\n`; for Inkling 2 Mini and Inkling 2 Small it is a one-sentence thought followed by the tokens with which Inkling opens a `tts_start` call, `<|message_model|><|content_thinking|>The user asked something; I answer it rather than repeat it.<|end_message|><|message_model|>tts_start<|content_invoke_tool_json|>{"name":"tts_start","args":{"text":"`, which also writes the thinking block, so no thinking is generated on this path.

| Thinking on (one / three sentences requested) | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| Low, TTS reads the model's text (old pipeline) | 890 / 1,263 ms | 1,265 / 1,579 ms | 2,192 / 2,637 ms |
| Low, `tts_start`, SGLang's parser | 1,191 / 1,866 ms | 1,828 / 2,403 ms | 3,304 / 3,124 ms |
| Low, `tts_start`, streamed argument | 1,168 / 1,872 ms | 1,870 / 2,252 ms | 3,308 / 2,717 ms |
| Medium, TTS reads the model's text (old pipeline) | 914 / 1,313 ms | 1,464 / 1,718 ms | 4,827 / 5,701 ms |
| Medium, `tts_start`, SGLang's parser | 1,455 / 2,033 ms | 2,027 / 2,923 ms | 3,536 / 3,783 ms |
| Medium, `tts_start`, streamed argument | 1,494 / 1,998 ms | 2,021 / 2,648 ms | 3,525 / 3,345 ms |

Every model has thinking on in this table, at the same named level: low and medium are `reasoning_effort` 0.2 and 0.7 for Inkling and `low` and `medium` for Qwen3.8-27B. There is no prefilled opening in it, because the opening starts the reply at the spoken text and leaves no room to think. With the prefilled opening no model generates thinking, and first audio arrives at:

| No thinking generated (one / three sentences requested) | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| `tts_start`, streamed argument and prefilled opening | 544 / 540 ms | 608 / 535 ms | 894 / 625 ms |

We replayed the same 15 clips per configuration and reply length; every turn produced audio (15 of 15 in each configuration and reply length). The table shows medians from the client sending the packet with the VAD-detected speech end to receiving the first nonempty audio chunk. On three-sentence requests at low thinking, streaming the argument saved 148 ms for Inkling 2 Small and 417 ms for Qwen3.8-27B (140 ms and 374 ms at medium; paired medians over the 15 clips). It saved Inkling 2 Mini nothing measurable (16 ms and 12 ms), because Mini writes its whole reply within about 70 ms of the first word, and it saved no model anything on one-sentence requests, where the argument is one sentence either way. With thinking on, the streamed tool path is still slower than direct text at low: by 712 ms, 673 ms and 161 ms on three-sentence requests for Inkling 2 Mini, Inkling 2 Small and Qwen3.8-27B (267 ms, 425 ms and 480 ms on one-sentence requests). Qwen3.8-27B's medians at medium say little on their own: its thinking takes from 1.4 s to 7 s depending on the question, on both paths, so the median of 15 clips falls between two groups of clips. This is the questions and not run-to-run noise, since a second run of its direct-text and streamed arms reproduced every median within about 200 ms. Paired by clip, its streamed tool path at medium was 115 ms faster than direct text on three-sentence requests and 209 ms faster on one-sentence requests, while both Inkling models were 590 ms to 1,015 ms slower.

**Failure.** Failed speech isn't retried; it returns `failed`. The profile's TTS timeout is 10 s instead of 120 s. If synthesis fails after some sentences have played, the response carries `heard_prefix` and a note telling the model to say only the rest. Without the note, the models started over from the first sentence in 231 of 238 samples; with it, in 2 of 238.

| Behavior (thinking low / medium) | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| First prompts calling `tts_start` under `tool_choice: required` (/80) | 80 / 80 | 80 / 80 | 80 / 80 |
| Plain questions answered with silence on a Session's first input (/80) | 0 / 0 | 0 / 0 | 3 / 9 |
| Inputs asking for no reply answered with silence (/80) | 73 / 74 | 74 / 75 | 80 / 78 |
| Spoke again after a completed speech (`silent` tool, /80) | 11 / 10 | 53 / 51 | 1 / 4 |
| Spoke again after a completed speech (`silent` action, /80) | 26 / 38 | 23 / 22 | 6 / 15 |
| Started over after a failure (no note, /40) | 39 / 38 | 39 / 39 | 38 / 38 |
| Started over after a failure (with note, /40) | 1 / 1 | 0 / 0 | 0 / 0 |

The counts in this table and in the thinking table come from one set of inputs, the same for every model: 40 spoken questions and 20 inputs that ask for no reply, sent as first prompts with the shipped system prompt and `tts_start` tool, thinking on, two samples each, and counted by rule. The `silent` tool and the `silent` action are not part of the shipped cascade, so we wrote each in two wordings and added the counts. Wording moved a count by up to 10 of 40; under either wording Qwen3.8-27B spoke again least and, with the `silent` tool, Inkling 2 Small most. One failure sample each of Inkling 2 Mini and Inkling 2 Small at medium had a one-sentence speech and is left out, so those two cells are out of 39.

## Evaluation

We ran [four evaluations](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/pull/311#issuecomment-7124364587) with Inkling 2 Mini, Inkling 2 Small and Qwen3.8-27B: Qwen3.8-27B on one H200, Inkling 2 Mini on one H200 and Inkling 2 Small on four H200s (`--tp 4`). The latency and Full-Duplex-Bench runs use [SGLang Omni at `21a6d22`](https://github.com/sgl-project/sglang-omni/tree/21a6d229c), which adds Inkling's streamed arguments and prefilled opening ([#4495](https://github.com/sgl-project/sglang-omni/pull/4495)); Inkling 2 Small runs behind the same chat service inside its own serving image. Every model runs with thinking on at the same named level: low and medium are `reasoning_effort` 0.2 and 0.7 for Inkling and `low` and `medium` for Qwen3.8-27B. The text gives the mean over the three models and the tables give each model. Latency is covered above.

**Full-Duplex-Bench.** We ran the direct-text and tool paths at four concurrent sessions on 100 v1.5 pairs (25 per category) and 30 v1.0 samples (10 per public subset), twice per configuration. Averaged over the three models, both thinking levels and both runs, the local Qwen judge labeled 19.5 of 25 direct-text backchannel cases as resuming the original speech, against 15.2 of 25 for the tool path. Cases labeled as responding went from 0.2 to 1.9 of 25, and v1.5 mean response latency from 1.79 s to 1.86 s. The licensed CANDOR subsets are left for later.

| Direct text → tool path, thinking on, two runs added | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| Low: backchannels judged as resuming the original speech (/50) | 46 → 29 | 48 → 31 | 30 → 26 |
| Low: cases judged as responding (/50) | 1 → 10 | 0 → 4 | 0 → 1 |
| Low: v1.5 mean response latency | 1.52 s → 1.57 s | 1.40 s → 1.38 s | 1.97 s → 2.21 s |
| Medium: backchannels judged as resuming the original speech (/50) | 43 → 31 | 48 → 37 | 19 → 28 |
| Medium: cases judged as responding (/50) | 0 → 4 | 1 → 3 | 0 → 1 |
| Medium: v1.5 mean response latency | 1.91 s → 2.11 s | 1.73 s → 1.60 s | 2.22 s → 2.28 s |

Every model has thinking on at the same named level on both paths, so the tool path here streams the argument without the prefilled opening. Both Inkling models resumed after fewer backchannels on the tool path in every run, by 4 to 10 of 25 cases, and their two runs agree within 3 cases. Qwen3.8-27B's counts are less stable: the same configuration moved by up to 6 cases between runs, and the judge left 16% to 52% of its backchannel cases uncertain, against 0% to 28% for the Inkling models. Its tool path resumed after 2 fewer cases than direct text at low in both runs and after 4 and 5 more at medium. Response latency repeated within about 0.1 s in every configuration.

**Fragments after repeated interruptions.** We rebuilt four 20-turn conversations from the topics in [#3255](https://github.com/sgl-project/sglang-omni/pull/3255#experiments), marked the first 30% of each prior reply as heard, and sampled 32 final answers per conversation and history format, 128 per format. With thinking on, no answer of any model ended in an ellipsis under any of the three history formats, at either level.

| History format (answers ending in an ellipsis, /128; thinking low / medium) | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| Trimmed reply with a system note | 0 / 0 | 0 / 0 | 0 / 0 |
| English inline note | 0 / 0 | 0 / 0 | 0 / 0 |
| `tts_start` with `heard_prefix` | 0 / 0 | 0 / 0 | 0 / 0 |

The fragments appear with thinking off: the same requests gave Qwen3.8-27B 16/128 (12.5%) for the trimmed reply and 0/128 for the other two formats (12/128 in the original run).

**History replay.** In both conditions the tool argument held a random code; the heard condition also had it in `heard_prefix`. Asked whether they had read the code, every model answered correctly in both conditions for all 32 pairs with `tts_start` and `heard_prefix`, at both thinking levels, and put the right code in its next `tts_start` in 58 to 64 of 64 cases. With a trimmed reply they were right in both conditions for 12 to 29 of the 32 pairs, and with an inline note for 5 to 11. These are direct chat requests with a supplied playback history: they test how the model uses the record, not live playback alignment.

| Measure (thinking low / medium) | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| `tts_start` with `heard_prefix`: correct in both conditions (/32 pairs) | 32 / 32 | 32 / 32 | 32 / 32 |
| Trimmed reply with a system note: correct in both conditions (/32 pairs) | 24 / 29 | 27 / 28 | 12 / 19 |
| English inline note: correct in both conditions (/32 pairs) | 11 / 8 | 10 / 11 | 5 / 8 |
| Right code in the next `tts_start` (/64) | 62 / 60 | 64 / 64 | 63 / 58 |

## Limitations and Open Questions

Most of what's open is about where control should sit.

**Thinking and tool calls.** We want thinking on, but users won't wait for it, and tool calls take time too. One idea is a short acknowledgment while the model thinks or calls a tool, such as "OK, let me check the weather" within half a second. The hard part is that this line and the speech after the tool returns are two separate syntheses, and they have to sound like one continuous utterance.

**Screen.** Someone driving an electric car should hear "72°F" and see the week on the touchscreen. We don't yet have a good answer for rendering: how should the model decide what to render? TTS as a tool has made what gets spoken fairly smart, and it lets what is shown differ from what is spoken; but the demo still shows exactly what it says, and what to render still needs design.

**More tools.** TTS is the only tool so far; Google search and weather lookups are next. Asynchronous tools conflict with "one speech per user input": "let me check, then tell you" needs a second speech. When a result arrives after the speech has ended, who triggers it: a system rule, or a fresh prompt to the model?

**Self-correction.** When should `tts_close` or an edit go back to the model? Only once the model can tell how far its speech has got.

**Steering.** When the user talks while the model is mid tool loop, should the system interrupt at once, or finish the loop and then add the user's words?

**Multiple speakers.** The ASR has no context and doesn't tell speakers apart, so several people's words merge into one. How far to support multiple speakers is still an open question.

**Interruption and speech rejection.** Interruption takes the fast path: the moment VAD detects voice, the Session closes the TTS. The VAD only reports whether someone is speaking, understands nothing of what is said, and its signal never reaches the LLM, so the model cannot judge whether it should stop. Its threshold is also low: background music, someone talking across the room or several people chatting all cut the speech off, while in the same noisy room the ASR often picks out the user's voice and leaves the music out. The system needs speech rejection: when the user says something unrelated to the conversation, or someone else talks, it should know not to stop. That is a semantic judgment, and it may cost some interruption latency. Directions include semantic probes on the VAD, a rejection model that accepts only the target speaker's voice, and a user setting for the VAD threshold.

The Session stops speech today because it sees continuous input and the playback timeline. If future models can use those signals directly, `tts_close` could become a model tool.

## Engineering Appendix

### Streaming Tool-Call Parsers

SGLang 0.5.23's `qwen3_coder` parser, used for Qwen3.8-27B, releases a string parameter only at its closing tag, so a reply's text arrives all at once. `sglang_omni.models.streaming_cascade.chat_service` is `sglang.launch_server` with a parser that subclasses SGLang's detector and releases string arguments as they are written. Because the chat service depends on SGLang internals, a test checks that it produces the same calls as SGLang's detector. Inkling 2 Mini and Inkling 2 Small use SGLang's `inkling2` parser, which also releases the text when the call ends; their chat service streams it the same way.

We timed the language model alone: 12 questions asked twice with `tool_choice: required` at temperature 0, medians from sending the request to the named piece of the `text` argument.

| Thinking low / medium | Inkling 2 Mini | Inkling 2 Small | Qwen3.8-27B |
|---|---:|---:|---:|
| SGLang's parser: the whole text arrives | 631 / 893 ms | 1,066 / 1,040 ms | 1,838 / 1,780 ms |
| Streamed argument: first text | 563 / 850 ms | 776 / 810 ms | 1,032 / 934 ms |
| Streamed argument: first sentence complete | 636 / 902 ms | 892 / 938 ms | 1,442 / 1,284 ms |

Streaming completes the first sentence about 400 to 500 ms earlier for Qwen3.8-27B and 100 to 175 ms earlier for Inkling 2 Small. Inkling 2 Mini gains nothing here: its replies to these questions are one sentence, written within about 70 ms of the first text. With the prefilled opening no thinking is generated, and the first text arrives after 36 ms, 46 ms and 66 ms and the first sentence is complete after 147 ms, 186 ms and 330 ms (Inkling 2 Mini, Inkling 2 Small, Qwen3.8-27B).

### Continuing the Call Opening

When a prompt requires a call, the `llm` node sends the call opening as an assistant message with `continue_final_message: true` and `tool_choice: auto`. The chat service puts the opening in front of the first streamed delta, so the reply still arrives as a streamed `tts_start` call. SGLang's own server would return the rest of the call as content, closing tags included, and the dialog would speak them, so this option needs the chat service. The opening skips the thinking, and that changes the reply. On the [24 question pairs of #4472](https://github.com/sgl-project/sglang-omni/pull/4472), against the unprefilled reply with thinking on (low / medium), the first sentence was the same in 4 / 4 pairs for Inkling 2 Mini, 4 / 3 for Inkling 2 Small and 6 / 6 for Qwen3.8-27B. Against Qwen3.8-27B with thinking off, where neither reply thinks, it was the same in 21 of 24.

### Computing `heard_prefix`

- The prefix ends at the last sentence boundary whose audio has finished playing.
- The playback position is the client's `truncate` report, capped at the audio the server actually sent.
- With the forced aligner, the position is word-accurate. A sentence cut before its audio was complete has no word timings, so the cascade's own recognizer transcribes the delivered audio and keeps the longest prefix of that sentence the transcript confirms.
- Later reports rewrite `heard_prefix`, so the next prompt holds the latest position.
- When a speech fails, the position comes from the synthesizer's `turn.done`, not `reply.progress`: with the aligner on, `reply.progress` reaches the dialog after `turn.done`, and its word positions can arrive after the sentence end.

This reuses the playback feedback path from the runtime post's appendix ("Committing Conversation History from Playback"). Only its destination, the tool response, is new.

## Usage and Acknowledgments

### Usage

The Inkling 2 Mini profile is `qwen3-asr-inkling-2-mini-vivian-tts-tool`; its chat service runs in its own environment, installed with `examples/streaming_cascade/inkling_chat_service.sh install`. Inkling 2 Small has no profile of its own yet; we ran it with a copy of the Mini profile whose LLM model is `thinkingmachines/Inkling-2-Small`, pointed at the same chat service started inside the Inkling 2 Small serving image. The Qwen3.8-27B profile is `qwen3-asr-qwen3.8-27b-vivian-tts-tool`; it needs two GPUs, or one H200 using the memory fractions given in the guide.

```bash
hf download Qwen/Qwen3-ASR-1.7B
hf download wilsonzheng0327/Qwen3-ASR-1.7B-dflash --revision d62bd4051e68282e02f93de774036a35f22f1373
hf download thinkingmachines/Inkling-2-Mini --revision b85c3cfa615fc413cc9af3379f491b963f3b586a  # Inkling 2 Mini
hf download thinkingmachines/Inkling-2-Small  # Inkling 2 Small
hf download Qwen/Qwen3.8-27B
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
hf download Qwen/Qwen3-ForcedAligner-0.6B
```

```bash
PYTHONPATH=. aligner-venv/bin/python examples/streaming_cascade/alignment_worker.py --model Qwen/Qwen3-ForcedAligner-0.6B --device cuda:1 --language English --port 8010
```

```bash
# Inkling 2 Mini
sgl-omni serve --config examples/streaming_cascade/profiles/qwen3-asr-inkling-2-mini-vivian-tts-tool/launch.yaml --enable-realtime --port 8765
# Inkling 2 Small: a copy of the Mini profile with the LLM model and base_url changed (no profile of its own yet)
sgl-omni serve --config inkling-2-small-tts-tool/launch.yaml --enable-realtime --port 8765
# Qwen3.8-27B
sgl-omni serve --config examples/streaming_cascade/profiles/qwen3-asr-qwen3.8-27b-vivian-tts-tool/launch.yaml --enable-realtime --port 8765
```

The "A live demo of the reference profile" section of the [Full-Duplex Guide](https://github.com/sgl-project/sglang-omni/blob/main/docs/developer_reference/full_duplex_guide.md), added in [#4473](https://github.com/sgl-project/sglang-omni/pull/4473), covers the aligner environment, the one-H200 layout and the browser demo. The "Speaking Through a Tool" section of the same guide describes the profile in full.

### Acknowledgments

The `tts_start` path landed in [sgl-project/sglang-omni#4428](https://github.com/sgl-project/sglang-omni/pull/4428). We thank everyone who contributed to it: Jingwen Gu, Yuhao Chen, Jiaxin Deng, Wenyao Gao, Jiadong Guo, Jeffero Qu, Junnan Li, Qun Lou, Jiaxian Luo, Jinyan Su, Xuesong Ye, Wilson Zheng, Chenyang Zhao.
