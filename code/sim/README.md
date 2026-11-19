# sim/: simulators

Two simulators. Both run the **real** robot code unchanged and replace only the world around it.

## Conversation scenarios (`world.py`, `scenarios.py`, `audio.py`)

```bash
python3 code/sim/run.py              # all 22 scenarios
python3 code/sim/run.py barge -v     # only names containing "barge", with the robot's log lines
```

A room with one clock, advanced in 32 ms audio chunks. The scenario says who is in view, who looks at the robot and
when, and who says what. The real ears (`mic.Ears`), barge-in and `conversation.py` run inside it. Synthetic voices
are speech-shaped noise at measured levels, the robot's own echo goes through the measured speaker → mic coupling,
and the mic has its own noise floor. Speech-to-text is a lookup by default (`--real-stt` sends the audio to the real
model, which costs money).

What the scenarios check, for example:
- nobody looks → nothing is ever sent to speech-to-text;
- Emre looks while Selim (a known voice) talks to his computer → dropped;
- the robot is talked over → it stops; if the transcript holds only its own words, it does not answer them;
- its own 4-sentence echo, with nobody looking → never an utterance, never a self-interrupt;
- a new face → the name is asked, the answer confirmed and stored, even when it comes 0.2 s after the question.

A scenario can carry a `known` diagnosis: a gap the sim found that is not fixed yet. It reports XFAIL while it still
fails and XPASS once it is fixed, so the list of known problems is executable.

## Rooms for navigation (`navworld.py`)

Random living rooms (3–4.5 × 2.5–3.5 m) with furniture, chairs on 3 cm legs, cables, a rug and people standing or
walking. The sim is deliberately unkind:

| Sensor | Model |
|---|---|
| speed | the real speed is 0.8–1.2× what the brain believes, with a first-order lag; spins coast |
| sonar | 24° cone 7 cm up; walls more than 40° off square give no echo; thin legs echo only near the beam centre; 3 % of pings lost |
| camera | the real `floor.py` ray code through the real ground geometry, with 1 % of cells wrongly "not floor" |
| people | two 6 cm legs; feet distance ±10 %, 10 % of detections missed |

Contacts are counted and must stay 0. Results: 24 rooms × 2 min exploring and 12 rooms × 3 min going to people gave
0 contacts and 0 cables touched. `code/nav.py --sim` shows the same room live on the navigation page.
