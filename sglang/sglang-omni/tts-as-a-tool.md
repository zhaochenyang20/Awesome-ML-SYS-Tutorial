# TTS as a Tool: The LLM Decides What to Say, the System Decides When to Stop

| Fragmented replies when every turn is interrupted | First prompts that call the speech tool | First audio of a multi-sentence reply |
|:---:|:---:|:---:|
| 25% → **[TBD]** | 12/24 → **36/36** | 2.48 s → **1.26 s** (old pipeline: 1.29 s) **[TBD: re-measure on one setup]** |

## TL;DR

In SGLang Omni's streaming cascade, TTS used to read every token the language model wrote. The model never chose to speak, never learned what the user had heard, and couldn't speak while doing anything else. We turned speech into a tool. The model calls `tts_start` with the text it wants said, and that call's single tool response reports whether the speech completed, was interrupted (along with the part the user heard), or failed. With Qwen3.8-27B, requiring the call, speaking once per user input, streaming the call's argument and prefilling its opening bring the first audio of a multi-sentence reply to 1.26 s. **[TBD: fragment-rate result.]** Asynchronous tools and correcting a speech mid-way are still open. The split of control is deliberate: the model decides what to say, and the system decides when speech stops and records what the user actually heard, because today's models can sense neither.

## 1. Introduction

Ask a human assistant to book you a flight and three things happen that nobody thinks about. They tell you the two options that matter rather than every fare on the screen. If you cut in halfway through the second option, they know you only heard the first. And they say "let me check" while they're already looking it up.

A full-duplex system has to do all of this while the conversation keeps moving. That takes two kinds of awareness a language model doesn't have: a continuous stream of input, and the passing of time. An LLM gets a prompt, writes a reply and stops. It doesn't know how far its words have got through the speaker, or how much the user heard before cutting in. Our [post on SGLang Omni's interactive runtime](https://www.sglang.io/blog/omni-interactive-runtime) ended on an open question: how much interaction behavior should a model learn, and how much should the surrounding system coordinate? Native full-duplex models learn timing inside the model. Cascaded systems hand it to an external controller, and in return each component can be upgraded on its own. This post answers that question for one action in a cascade: speaking.

Our answer is a split of control. The model decides what to say. The system decides when speech must stop, and keeps the record of what the user actually heard. The interface between the two is one that agents already know well: a tool call.

## 2. The Problem: TTS as a Fixed Stage of the Pipeline

Until this change, the streaming cascade ran ASR → LLM → TTS, with the synthesizer reading every token the model wrote. That had three consequences.

**The model says too much.** The cascade already strips Markdown, emoji and links before synthesis, so stray symbols aren't the issue. The issue is that an answer written for a screen is the wrong shape for speech: long, structured, thorough. Because everything the model writes gets spoken, it has no way to write one thing and say another.

**The model doesn't know what the user heard.** When the user cuts in, the cascade trims the reply in the history down to the part that was played, ends it with "…" and adds a system note. The model then reads its own chopped-off replies as examples of how it answers, and copies them. In [#3255](https://github.com/sgl-project/sglang-omni/pull/3255), in 20-turn conversations where every reply was cut, 25% of Qwen3.8-27B's replies came out as fragments such as "杜甫是唐代伟大的…" ("Du Fu was one of the great Tang…"), with the note in place.

**Speech can't run alongside anything else.** An assistant should be able to say "let me check" while a lookup runs, or ask for a slower, warmer delivery. When speech is simply what comes out of the pipeline, there's nowhere to start other work in parallel and nowhere to put speaking parameters.

All three problems have the same cause: speaking isn't something the model does. A speech has no identity and no result, and it leaves no record of what was done and how it turned out. A tool call provides all three. The `tool_call_id` identifies the action, the tool response is its result, and the call and its response together are the record in the history.

## 3. Design

The design rests on three constraints, each of which rules out a specific failure:

- **The Session is the only thing that changes state.** The LLM issues actions and the TTS carries them out, but only the Session decides whether speech is on. Nodes don't message each other directly, so their views of the state can't drift apart.
- **Each action has one owner.** `tts_start` belongs to the LLM and `tts_close` to the Session. Two parties never operate the TTS at once, so they can't race over who stopped what.
- **Each call gets one answer.** One `tts_start` gets one tool response. A call answered twice would need a second ID to tie the answers together. With one answer, the `tool_call_id` is all we need.

On top of these there is one division of labor. The model owns what is said, and the system owns stopping speech and recording what the user heard.

| Action | Called by | What it does |
|---|---|---|
| `tts_start(text)` | The LLM | Splits the text into sentences, then synthesizes and plays them one by one |
| `tts_close` | The Session, when the user starts talking | Stops playback at once and clears the buffers. It is not in the LLM's tool list. |

A Session plays one speech at a time. A second `tts_start` while one is playing gets `BUSY` rather than silently replacing it. The normal path and the interruption path look like this:

<div align="center">
  <img src="images/tts-tool-sequence.png" alt="A tts_start call: the LLM calls tts_start and the Session has the TTS speak; the speech either plays out and is recorded as completed, or the user starts speaking and the Session sends tts_close to the TTS and stop to the LLM, recording interrupted with heard_prefix" width="90%">
</div>

Every `tts_start` ends in exactly one outcome:

<div align="center">
  <img src="images/tts-tool-outcomes.png" alt="Outcomes of tts_start: a refused call or one with no words to speak returns failed; an accepted speech ends as completed, interrupted with heard_prefix, or failed with heard_prefix" width="90%">
</div>

`heard_prefix` is the part of the text the user actually heard. Section 8.3 explains how it is computed.

**Interruption takes a fast path.** When voice activity detection hears the user, the Session closes the TTS and stops the LLM at once, without asking the LLM. Waiting for the user's words to go through ASR and the LLM first would leave time for more audio to play. Everything played in that window would be recorded as said, although the user, who was talking, never heard it.

## 4. Design Trade-offs

Each trade-off below follows the same pattern: the question, the options, what we chose, why, and what it costs.

### 4.1 When Does a Call Get Its Answer?

A speech lasts seconds, and the model needs to know how it ended. The obvious design answers `tts_start` immediately with "accepted" and reports completion later as a separate event. But then every call has two answers, and something has to tie them together, which means a second ID next to the `tool_call_id`, and every later operation has to carry it.

We answer only when the model needs to act on the answer. A refused call is answered at once. An accepted call is answered once, when the speech ends. This keeps the ordinary meaning of a tool call, one call and one result, and the second ID disappears along with everything that would have depended on it. The cost is that the model learns nothing while it is speaking. That costs little, because nothing prompts the model while it speaks anyway.

### 4.2 Who May Stop a Speech?

We considered letting the LLM close its own speech, a `tts_edit` that replaces sentences in flight, letting a new `tts_start` replace the speech in progress, and queueing concurrent starts. We chose something narrower. Only voice activity detection triggers a close, and it goes through the Session. The first version has no LLM close and no edit, and a second start gets `BUSY`.

The reason goes back to the thesis. The LLM doesn't know how far its speech has got, and it hasn't been trained on an interface like this, so we can't expect it to stop at the right moment. An edit needs sentence splitting, lock points, version numbers and a way to discard audio that is already out of date. A close already returns the stop position, so closing and starting again covers the same need with far less machinery. The cost is real: the model can't correct itself in the middle of a speech.

### 4.3 After an Interruption

**How the model learns what was heard.** We could stream playback progress back to the model, or give it a list of sentences to query. Instead, the stop position is reported once, at the interruption, in the tool response. An LLM has no sense of time and usually assumes it finished speaking, so feeding it progress while it generates wouldn't improve its decisions. The position is accurate to the sentence by default and to the word with an aligner. We don't try to align to tokens. The model isn't rolled back either. Its next prompt shows the new input next to the interrupted call's result.

**How an interrupted speech appears in the history.** There were three candidates:

| Approach | What goes wrong |
|---|---|
| Trim the reply to what was heard (the old cascade) | The model imitates the trimmed replies: 25% fragments when every turn is cut |
| Insert "the user interrupted you here" at the cut | Weaker models start saying lines like this themselves |
| Keep the full call and put `heard_prefix` in the tool response (ours) | Nothing the model wrote is rewritten |

The third approach keeps the model's intent and the outcome separate. The call holds the full text the model meant to say, and the response says how much of it got through. The history never contains a chopped-off reply for the model to copy.

## 5. Making a Real Model Use It

The design was the easy part. Getting Qwen3.8-27B to use it well took a series of experiments.

**Forcing the call.** With `tts_start` offered as an option, only 12 of 24 first prompts called it, and the rest answered in plain text. With `tool_choice: required`, 36 of 36 did. With thinking on, the model calls the tool without being forced, but writing the whole text then took a median of 3.1 s, so the profile runs with thinking off. As a fallback, if a server ignores `tool_choice` and the model writes plain text, that text is spoken as if it were a `tts_start` call, and its result is reported, cut short or followed up in the same way.

**Ending a turn.** If every prompt requires a call, the model can't choose silence. So how does a turn end? We tried giving it a way out, a `silent` tool. On a Session's first input it chose silence for 34 of 50 plain questions. After a completed speech it repeated its answer in 10 of 49 samples with a `silent` tool, and in 22 of 50 with a `silent` action. So ending a turn became a system rule: **one speech per user input**. After a completed speech the model isn't prompted again, and the result goes into its next prompt. It is prompted again only if its call was refused or the speech failed, at most four times per input. The cost is a conflict with the most typical use of an asynchronous tool: say "let me check", then speak again when the result arrives. Section 7 comes back to this.

**First-audio latency.** This came down in three steps:

1. A tool call's arguments normally arrive only once the call is complete. SGLang's `qwen3_coder` parser releases a parameter only at its closing tag, so the model had to write its whole answer before the first word was spoken.
2. Our chat service streams the `text` argument as the model writes it, so the first sentence is synthesized while the model is still writing the rest.
3. Every call starts with the same 14 tokens, `<tool_call>\n<function=tts_start>\n<parameter=text>\n`. We send them as an assistant message the model continues (`continue_final_message: true`), so decoding starts at the spoken text. The model's first character came after a median of 77 ms instead of 293 ms. One detail: these requests use `tool_choice: auto`, not `required`. The opening already commits the model to the call, and the grammar that `required` imposes would make it open a second call.

| Configuration | Multi-sentence reply | One-sentence reply |
|---|---|---|
| TTS reads the model's text (old pipeline) | 1.29 s | 1.14 s |
| `tts_start`, SGLang's parser | 2.48 s | 1.38 s |
| `tts_start`, streamed argument | 1.47 s | 1.39 s |
| `tts_start`, streamed argument + prefilled opening | 1.26 s | **[TBD]** |

*Time from the end of the user's speech to first audio, median. The first three rows were measured on two H200s over 15 turns each. The last row was measured separately, over 9 questions asked twice each. **[TBD: re-measure all four rows on one setup.]***

A one-sentence reply gains nothing from streaming, because its first sentence ends where the call does.

**Failure.** Failed speech isn't retried. It fails loudly and returns `failed`. In this profile the TTS timeout is 10 s instead of 120 s. When synthesis fails after some sentences have played, the response carries `heard_prefix` along with a note telling the model to say only the rest. Without the note, the model restarted from the beginning in 13 of 20 samples. With it, that dropped to 0 of 20.

## 6. Evaluation

**[TBD: this section waits on four runs.]** The numbers above are engineering checks on small samples: 24, 36, 50 and 20 runs. As in the runtime post, 25 samples per category carry roughly 20 points of sampling noise, so every number is reported with its sample size.

- **Latency on one setup.** The old pipeline, `tts_start` with SGLang's parser, the streamed argument, and the streamed argument with the prefilled opening, all on the same hardware, for one-sentence and multi-sentence replies. **[TBD]**
- **Full-Duplex-Bench.** The `tts-tool` profile against the old text path on the same revision and hardware: 25 pairs per v1.5 category and 10 samples per v1.0 subset. **[TBD]**
- **Fragment rate when every turn is interrupted.** This uses the setup from #3255: four 20-turn conversations, with the first 30% of each reply counted as heard and 32 samples per cell. Trimmed replies with a note give 25%. The full call with `heard_prefix` gives **[TBD]**.
- **History replay.** We plant a random detail in the unplayed tail of a speech, then ask about it on the next turn, and measure how often the model believes it already told the user. **[TBD]**

## 7. Limitations and Open Questions

What's left open is mostly a question of where control should sit.

**Asynchronous tools.** "One speech per user input" conflicts with "let me check, then tell you". When a weather result arrives after the speech has ended, who triggers the next speech: a system rule, or a fresh prompt to the model?

**Self-correction.** When should `tts_close` or an edit go back to the model? That only makes sense once the model can tell how far its speech has got.

**Steering.** When the user talks while the model is in the middle of a tool loop, should the system interrupt immediately, or let the loop finish and then add the user's words?

Today we keep stopping speech and recording what was heard in the system, because the model can't sense time. As models gain that sense, the line between model and system should move toward the model. The interface wouldn't need to change: the model would simply find `tts_close` in its tool list.

## 8. Engineering Appendix

### 8.1 A Streaming `qwen3_coder` Parser

SGLang 0.5.23's `qwen3_coder` parser releases a string parameter only at its closing tag. With it, the text of a reply arrived all at once, 1.05 s into the reply. `sglang_omni.models.streaming_cascade.chat_service` is `sglang.launch_server` with a parser that subclasses SGLang's detector and releases a string argument as the model writes it. With that parser, the first sentence was complete at 0.59 s. A test checks that it produces the same calls as SGLang's detector, because the chat service depends on SGLang's internals.

### 8.2 Continuing the Call Opening

When a prompt requires a call, the `llm` node sends the 14-token opening as an assistant message with `continue_final_message: true` and `tool_choice: auto`. The chat service puts the continued opening in front of the first streamed delta, so the reply still arrives as a streamed `tts_start` call. SGLang's own server would return the rest of the call as content, closing tags included, and the dialog would speak them, so this option needs the chat service. With the opening, the first sentence matched the reply without it in 21 of 24 pairs.

### 8.3 Computing `heard_prefix`

- The prefix ends at the last sentence boundary whose audio has finished playing.
- The playback position is the client's `truncate` report, capped at the audio the server actually sent. A client report never moves the position past what was delivered.
- With the forced aligner, the position is accurate to the word. A sentence cut before its audio was complete has no word timings, so the cascade's own recognizer transcribes the delivered audio and keeps the longest prefix of that sentence the transcript confirms.
- Reports of what was heard that arrive later rewrite `heard_prefix`, so the next prompt holds the latest position.
- When a speech fails, the position comes from the synthesizer's `turn.done`, not from `reply.progress`. With the aligner on, `reply.progress` reaches the dialog after that `turn.done`, and its word positions can arrive after the sentence end.

None of this is new machinery. The tool response reuses the playback feedback path described in the runtime post's appendix ("Committing Conversation History from Playback"). The only change is that its result now goes into the tool response.

## 9. Usage and Acknowledgments

### Usage

The profile is `qwen3-asr-qwen3.8-27b-vivian-tts-tool`. It needs two GPUs, or one H200 using the memory fractions given in the guide.

```bash
hf download Qwen/Qwen3-ASR-1.7B
hf download wilsonzheng0327/Qwen3-ASR-1.7B-dflash --revision d62bd4051e68282e02f93de774036a35f22f1373
hf download Qwen/Qwen3.8-27B
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
hf download Qwen/Qwen3-ForcedAligner-0.6B
```

```bash
PYTHONPATH=. aligner-venv/bin/python examples/streaming_cascade/alignment_worker.py --model Qwen/Qwen3-ForcedAligner-0.6B --device cuda:1 --language English --port 8010
```

```bash
sgl-omni serve --config examples/streaming_cascade/profiles/qwen3-asr-qwen3.8-27b-vivian-tts-tool/launch.yaml --enable-realtime --port 8765
```

The "A live demo of the reference profile" section of the [Full-Duplex Guide](https://github.com/sgl-project/sglang-omni/blob/main/docs/developer_reference/full_duplex_guide.md), added in [#4473](https://github.com/sgl-project/sglang-omni/pull/4473), covers the aligner environment, the one-H200 layout and the browser demo. The "Speaking Through a Tool" section of the same guide describes the profile in full.

### Acknowledgments

The `tts_start` path landed in [sgl-project/sglang-omni#4428](https://github.com/sgl-project/sglang-omni/pull/4428). We thank everyone who contributed to it: Jingwen Gu, Yuhao Chen, Jiaxin Deng, Wenyao Gao, Jiadong Guo, Jeffero Qu, Junnan Li, Qun Lou, Jiaxian Luo, Jinyan Su, Xuesong Ye, Wilson Zheng, Chenyang Zhao.
