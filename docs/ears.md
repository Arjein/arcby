# Arcby - ears (speech in, Turkish)

How the robot hears. Code: `code/robot/vad.py` (is someone speaking: Silero VAD v5, MIT, ONNX, ~0.1 ms per 32 ms),
`code/robot/mic.py` (the I2S mic as a stream of utterances: pre-roll 0.3 s, ends after 0.45 s of silence, muted
while the robot speaks, nothing written to disk), `code/robot/stt.py` (speech -> text: Whisper on the Pi or Gemini in
the cloud), test tool `code/hwtest/earshots.py`.

## Listening rule (planned)
Only when the robot itself started a conversation (asking a name) or someone FACING it speaks - never the room in
general. Vision pauses while it listens (the demo showed Whisper starved for 50 s by the camera work on 4 cores).

## First measurement (2026-11-01): the robot's own TTS phrases
45 cached Gemini-TTS greetings in the Charon voice with known text ("Günaydın Selim, uyandın mı?", "Merhaba Doğu!"),
straight from the files - clean speech, no mic, no room: an UPPER bound. WER = word error rate (Turkish-normalised).

| backend | WER | names right | s / utterance | where |
|---|---|---|---|---|
| Whisper tiny | 43 % | 18/45 | 2.0 | Pi |
| Whisper base | 46 % | 19/45 | 4.0 | Pi |
| Whisper small | 24 % | 29/45 | 12.6 | Pi (7.7x real time: too slow) |
| Gemini 3.5 Flash-Lite | 17 % | 31/45 | 1.1 | cloud |
| Gemini 3.8 Flash | 12 % | 37/45 | 3.2 | cloud |
| **with the family names as a hint** ("Selim, Emre, Meltem, Halil, Doğu, Arkbi") | | | | |
| Whisper tiny | 30 % | 36/45 | 4.5* | Pi |
| Gemini 3.5 Flash-Lite | 9 % | 44/45 | 1.1 | cloud |
| **Gemini 3.8 Flash** | **0 %** | **45/45** | 3.1 | cloud |

\* alive.py was running at the same time. Most errors are names ("Merhaba Emre" -> "Merhabalar"). `gemini-2.5-flash`
is "no longer available to new users" -> `gemini-3.8-flash`. Next: the same on Selim's real voice at 1 / 2 m
(`earshots.py session`, then `compare`); Gemini only if Selim agrees that clips of people talking TO the robot may go
to Google (paid tier: not used for training).

## Selim's real voice (2026-11-01, `earshots.py session` + `compare`)
15 sentences at 1 m and 2 m through the robot's mic (high-pass + auto gain + VAD): 28 of 30 caught ("Dur" at 1 m
too short, "Arkbi" at 2 m while walking). All with the family names as a hint.

| backend | WER | names right | s / utterance |
|---|---|---|---|
| Whisper tiny | 65 % | 10/17 | 2.1 |
| Whisper small | 29 % | 14/17 | 12.3 |
| Gemini 3.5 Flash-Lite | 17 % | 15/17 | 1.3 |
| **Gemini 3.8 Flash** | **0 %** | **17/17** | 2.9 |

Flash-Lite got 8 of 28 wrong, some changing the meaning: "Hayır" -> "hayırsız", "Bugün hava nasıl" -> "Bugün alan
asalım", "Arkbi, saat kaç" -> "Ark bir saat kaç?". 3.8 Flash: none wrong.

The mic signal is 99 % rumble / drift below 80 Hz and the speech is very quiet: without the 100 Hz high-pass the VAD
found no speech at all (max 0.18), with it 0.98; the auto gain runs near its x120 limit between sentences.
The gain must not learn from the robot's own voice (it did at first and stayed too low for seconds).

## Who is speaking: voice recognition (2026-11-01, `robot/voices.py`)
Local only (no audio leaves the robot, none is stored): an ONNX speaker model turns an utterance (80 log-mel fbank,
Kaldi-style, mean-normalised) into a 192-d voice print. A person's voice = up to 40 prints; an utterance is theirs when
its 3 closest prints average >= 0.5 and beat the next person by 0.08. Prints are learnt only from sentences whose
speaker the robot SAW (exactly one known face looking as the sentence started, voice not clearly someone else's,
>= 0.8 s); `people_cli forget` deletes them (by hand only - the robot has no "forget me" by voice since 2026-11-01, Selim). Selim's were seeded from his 27 labelled session clips.

Bake-off on the Pi: Selim's 28 robot-mic clips (quiet, x50-120 auto gain, ~1.5 s) against 34 other voices (Charon TTS,
three speakers of the sherpa-onnx test clips). Check on the clean test clips first - which feature setting each
export needs (same speaker / different speakers, cosine):

| model | mean-normalised | same speaker | different, max | on Selim's clips: recognised / others taken for him (at 0.5, 3 closest prints) | ms per s of speech |
|---|---|---|---|---|---|
| **3D-Speaker CAM++ zh-en advanced** | yes | 0.72-0.83 | 0.45 | **79 % / 0 %** (0.45: 82 % / 0 %) | 230 |
| WeSpeaker ResNet34-LM | yes | 0.62-0.73 | 0.37 | 46 % / 3 % | 690 |
| 3D-Speaker ERes2Net (VoxCeleb) | yes | 0.60-0.76 | 0.44 | 32 % / 3 % | 610 |
| WeSpeaker CAM++ (VoxCeleb) | either | 0.01-0.96 | 0.86 | broken export / features | 180 |

Lessons: (1) WeSpeaker ResNet34 WITHOUT mean normalisation scored Selim-vs-Selim 0.90 but everything ~0.8: it was
matching the microphone, not the voice - a test where all the target's clips come from one mic and all the others
from other recordings rewards that. (2) Cutting the clips to VAD speech did not help. (3) Selim's clips score only
~0.33 pair to pair (clean speech: ~0.8): quiet speech through a gain of x50+ is hard; more prints per person help.
Not measured yet: Selim vs Emre through the same mic (brothers) - VOICE_MARGIN is the guard until then.

## The head turns to voices (robot/attention.py)
Every utterance (also those never sent anywhere) gives a "voice" event: the speaker's face if in view (followed,
x3 interest for 4 s); else where that person was last seen (orient, 2 s, then a search); else - if nobody in view
looks at the robot - a look round the room (scan, +-50 deg, 3.5 s; the first face that appears is taken). The same
voice turns it at most every 8 s; its name ("Arkbi" in the transcript, "called") always does.

## Bugs found on the robot (2026-11-01 night, battery + head)
- **The VAD drifts deaf.** Silero v5 keeps a state between chunks; after ~1 min of loud (x120) room noise it never
  said speech again (the same 15 s of Selim talking: 107 speech chunks on pass 1, 52 on pass 3, 0 from pass 6; 102
  after `reset()`). alive.py "did not hear" Selim for ~25 min. Now reset after 4 s without speech and after each
  utterance (20 passes: ~100 every pass). The status line shows `ears N heard, gain, level, VAD max, chunks/s`.
- **The gain must learn from speech only** (a click / servo noise took it from x120 to x6 for 20 s) - but then it
  waits at x120 between sentences and the first syllables clipped ("napıyorsun" -> "Dua etsin"). Utterances are now
  cut from the un-gained sound and normalised once (peak -6 dBFS); only the VAD sees the gained signal.
- **Name hints make Gemini guess names** for unclear sound ("beni duyuyor musun" -> "Meltem gelsin"): the hint now
  says to write a name only when clearly spoken.
- ~~Room background above 100 Hz ~-65 dBFS~~ - that was mostly our own code (review 2026-11-01, below). The real
  background is ~-80 dBFS (a 5 kHz whine comes and goes - not the speaker, servos, fan or HC-SR04); Selim's quiet
  speech ~-60 dBFS at 1 m. The mic port still faces up through the deck.

## Review fixes (2026-11-01, replayed on the Mac)
Replay = the real `Ears` loop + Silero on the silent tail of `~/scratch/hwtest/listen/heard.wav` (mirror-looped) with
Selim's 28 session clips (`earshots/20261101-015715`) mixed in at a set level, one every 6 s (quiet) or 30 s (loud).

| speech level | before | after |
|---|---|---|
| -70 dBFS (quiet, far) | 15 / 28 | 28 / 28 |
| -66 dBFS | 23 / 28 | 28 / 28 |
| -60 dBFS, 4x background | 8 / 28 | 27 / 28 |
| -40 dBFS (normal, close) | 18 / 28 | 25 / 28 |
| -30 dBFS (loud) | 11 / 28, first syllable clipped | 25 / 28 |
| -25 dBFS | 7 / 28, first syllable clipped | 26 / 28 |

No false utterances in silence before or after; the utterance now starts ~0.15-0.25 s before the speech again.
- **A self-made click train.** `to_16k_mono` subtracted each 32 ms chunk's own mean: the mic's slow drift became a
  step at every chunk border, 31 times a second, louder than quiet speech (pure drift came out at -56 dBFS instead of
  ~-90). Gone - the high-pass removes DC and drift smoothly across chunks.
- **Loud speech was never speech.** At x120 normal speech was hard-clipped to a square wave, the VAD scored it low,
  so the gain (which learns from speech only) never came down: a deadlock. Now a per-chunk peak limiter (never a
  clip), and a full second the limiter had to halve the sound teaches the gain too (a click or servo does not).
- **The VAD reset could be held off** by noise scoring 0.35-0.5 (a TV): it is now reset every 4 s while nothing is
  being heard, whatever the score.
- **"Dur" was too short** (5-7 speech chunks, the minimum was 8): `MIC_MIN_S` 0.25 -> 0.16 s.
- **The robot's own last syllable:** the ears stay deaf `MIC_UNMUTE_HOLD_S` (1 s) after the voice is done (up to
  ~0.7 s of it is still in the speaker's buffers then) and nothing heard meanwhile becomes pre-roll.
- **Timing:** chunk times come from the sample count (read times jump when the thread catches up); each utterance
  has `onset` = its first speech chunk (t0 is the pre-roll start, ~0.25 s earlier) for the listening rule.
- 48 -> 16 kHz with a 63-tap low-pass (the mean of 3 samples folded 8-16 kHz into the speech band at up to 0.67);
  the tail after an utterance is the next one's pre-roll; arecord is started again (1 s, 2 s, ... 60 s) when it
  ends instead of the ears thread just stopping (status: "N mic restarts").
- STT: the prompt says to write each word in the language it was spoken (never translate); answers with no letters
  or only sound descriptions ("...", "(anlaşılmıyor)", "[müzik]") count as nothing; a dead Whisper server is started
  again. Gemini's thinking then stayed at the model default (with it off, 3.8 Flash erred like Lite on the TTS
  phrases) - turned off on 2026-11-01 for speed, see below.

## Faster replies (2026-11-01)
The end of a sentence to the robot's voice took 10-25 s on the robot: end silence 0.6 s, voice-ID ~0.4 s BEFORE the
STT, Gemini 3.8 Flash STT 2-9 s (default thinking), Claude Haiku 5.5 1-4 s for the whole reply (up to 3 long
sentences), Gemini TTS ~3-4 s per reply (the primary model out of its daily quota: a 429 first, then the fallback),
then playback. Changed:

| stage | before | now |
|---|---|---|
| end silence (`MIC_END_S`) | 0.6 s | 0.45 s - Selim's 28 session sentences stay whole down to 0.35 s (longest pause inside one: 0.1 s), "Dur" kept |
| voice-ID | before the STT (+0.4 s) | next to it (two threads); only "who speaks" waits for it - and, when nobody looked and only the voice can tell whether it is the awaited answer, the decision to send it at all |
| STT | default thinking, 2-9 s | **live** (`robot/live_stt.py`, `gemini-3.5-transcribe-live` over a websocket, `robot/wsock.py`): the sentence streams while it is said - only once the looking rule allows it (looked at within `CONV_GAZE_START_S` of the onset, or a conversation open), else nothing is sent; our VAD sends activityStart / activityEnd (the model's own skipped quiet speech), `languageCodes` tr-TR + en-US (without: Hindi / English for quiet Turkish). Words +0.3-0.5 s after the end (Pi, 2026-11-01). Live empty / failed / not in `STT_LIVE_WAIT_S` -> the clip STT (3.8 Flash, `thinkingLevel: low` - it refuses "minimal", and without a setting it thought at its default: 3-5 s; low ~2.8 s) |
| mind | whole reply, 1-4 s | streamed (`messages.stream`): sentence 1 goes to the voice as soon as it is complete (~0.5-1 s); the prompt asks for a short first sentence; the stream is closed after `MIND_MAX_SENTENCES` |
| TTS | one call per reply, then play | per sentence, pipelined: sentence 2 is made while 1 plays (`Voice.hold()` keeps the ears muted between them). **Streamed** (`gemini-3.8-flash-tts`, Interactions API, `stream: true`): playing starts with the first piece ~2.1 s after the call (whole phrase ~4 s), the rest comes faster than it plays; `CardStream` sets one gain (`TTS_STREAM_PEAK`) and resamples without seams; cached when whole. A model refused with 429 twice in a row rests until Google's midnight (`TTS_DAY_AFTER_429`): no ~0.4 s refusal in front of every reply |
| filler | - | when sentence 1 is not cached, `CONV_FILLER_P` (70 %) of the time a short cached filler plays at once ("Hmm.", "Hmm, bakalım.", "Şey...", "<breath> Bir saniye.", "Bir bakayım."; English ones for English) - never the same twice in a row, never one that is not cached (made at start-up by `Voice.prepare`) |

Vocal tags: Gemini 3.8 TTS makes real sounds from inline `<tags>` (English, also inside Turkish): `<chuckle>`,
`<sigh>`, `<breath>`, `<short pause>`... (`tts.AUDIO_TAGS` = Google's list). The mind is offered the calm ones
(`mind.MIND_TAGS`) and uses at most one, only exactly where a person would make the sound; old `[square]` names map
onto them (`[uhm]` -> `<breath>`), any other bracketed text is removed (nothing odd read out), and the 2.5 fallback
model gets none. Selim by ear 2026-11-01: "tags pretty good, no wrong pronunciation". `TTS_AUDIO_TAGS = False` strips
them all.

Every answered utterance logs one line in `alive.log`, seconds after the end of speech:
`timing (s after end of speech): utterance in +0.47, stt done +1.6, voice-id +0.9, mind 1st sentence +2.3, 1st audio +5.2 (gemini), playing +5.2, filler 'Hmm.'`

Expected (from the measured parts): first sound (filler) ~2.5-3 s after the end of speech, the answer itself ~4.5-6 s
(was 10-25 s): 0.45 end silence + ~1.2 STT + ~0.7 mind sentence 1 + ~2.5-3 TTS of a short sentence. The TTS is now
the biggest part; streaming TTS would be the next step.
Not measured yet: STT accuracy with thinking off on Selim's real voice (the TTS-phrase test erred like Lite) - check
the `heard` lines.

## Barge-in: talking over the robot (2026-11-01)

Someone talks while the robot speaks -> it stops within ~0.3 s and listens (`robot/bargein.py`, `BARGE_*`).
Its own voice reaches the mic as loud as a person 1-2 m away, and the small speaker + body distort it (a waveform
echo canceller took out only 6-8 dB; coherence speaker -> mic ~0.15). What works: `EchoRef` keeps what speakerd
plays and when (predicted on send, corrected by its "done" times); `BargeIn` predicts the echo's power in 9 bands
(`BARGE_COUPLING_DB`: the speaker is -42 dB below 250 Hz, -6 dB at 1.8-2.6 kHz; the room's tail fades
`BARGE_DECAY_DB` per 32 ms, `BARGE_LAG_S` = 108 ms) and calls it someone else when 3+ bands are `BARGE_THRESHOLD_DB`
over it in 5 of 8 chunks. The coupling keeps learning while the robot speaks alone; the room's floor while it is
silent. Only when someone looked at the robot in the last `BARGE_LOOK_S` or a conversation is open (a knock in a
pause of its sentence was the one false alarm on the test recording). Then: speakerd STOP (10 ms fade, the card keeps
running - no pop), `Voice.interrupt()` drops what is queued / still being made, the mind's next sentences are not said,
and the ears listen at once with the last `MIC_PREROLL_S` as pre-roll. Recording (Selim over the 4th of 4 sentences):
caught 0.1 s after he started.

## Barge-in that tunes itself (2026-11-02)

On the robot every reply was cut ~0.7 s in, also at speaker level 0.2: the damaged speaker's distorted echo was far
over `BARGE_COUPLING_DB` (fitted before it was damaged). Worse, after each cut the ears heard the end of its own
voice and answered it: "Merhaba Selim!" (the greeter) came back as Selim's "Merhaba Selim.", "Anneni çağıramam" as
"çağırma." - a loop of cut replies (alive.log 15:37-15:38).

- **Its own voice is not a voice print.** CAM++ prints of its echo in the mic are 0.19 alike to its clean TTS voice
  (median; people 0.22, clean vs clean 0.60) - the speaker and the room change it too much. What it knows exactly is
  what it played and when (`EchoRef`), so that is what the fixes use.
- **Its own words heard back after a cut are never answered** (`conversation.own_echo`: every word is one it said
  in the last 15 s - also the greeter's, `Voice.spoken` - or nearly: difflib >= `CONV_ECHO_FUZZ`, substrings from 3
  letters). Event `own_echo` (dashboard: "its own words heard back"). It has stopped already, which is all an
  interrupting person wanted; scenario 11a changed from "Efendim?" to no reply.
- **A false alarm teaches the detector**: if that sound also ended within `CONV_ECHO_END_S` of the cut (its voice
  fades by ~0.5 s; a person who talked over it goes on), `BargeIn.false_alarm()` raises each band that fired by how
  far it was over the threshold + 1 dB (at most `BARGE_FALSE_STEP_DB`, never over `BARGE_COUPLING_MAX_DB`) and saves
  it to `BARGE_TUNE_FILE` (`~/.local/share/arcby/barge.json`, read at start). Test: an echo 20 dB over the
  prediction stops cutting after a few false alarms, someone talking over it is still caught.
- **Calibration per speaker**: `python3 code/hwtest/bargecal.py` (alive.py stopped, nobody talking, ~30 s): the room
  for 2 s, then 11 cached phrases at the dashboard level; per band the 97th percentile of mic - played (+1 dB) is
  the coupling; the echo lag is re-measured; the detector is replayed on the recording (false barge-ins before ->
  after) and the result written to `BARGE_TUNE_FILE`. **Run it once after fitting the new speaker.**
- Snippet / heard WAVs were named HHMMSS and pruned by name: every file of a new day sorted before yesterday's and was
  deleted at once (no snippets of the 15:37 cuts). Now `YYYYMMDD-HHMMSS` and pruned by time written (`bargein.prune`).
