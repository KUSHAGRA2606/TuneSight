# TuneSight: a jal tarang that sees and listens

**Course:** DA321 Multimodal Data Processing and Learning, IIT Guwahati

**In two lines:** The camera measures how much water is in each identical-looking bowl, and the microphone measures the pitch of each strike.
A physics law, f = f₀ / √(1 + β(h/H)ⁿ), links the two. It is learned live from a few taps, which lets the system (1) tell which identical bowl was struck, (2) tune the bowls to a raga by sight, and (3) write the sargam notation as you play.

---

## 1. Why this idea is novel: what the literature search found and how we changed the idea

| Already published (so it is **not** our claim) | What we do differently |
|---|---|
| Material recognition from image + impact sound (ObjectFolder benchmark, CVPR 2023; Maleki et al. 2024; *Objects as Audio-Visual Modal Sound Fields*, ECCV 2026; Zhang et al. "Shape and material from sound", NeurIPS 2017) | We dropped material recognition as a claim. We use the same audio physics (modes = poles) only as a building block. |
| Liquid height from **pouring** sound (PouringNet, IROS 2019; audio-visual pouring networks, Wilson et al. 2019) | We use the **struck-bowl resonance** (not pouring noise), a physical law instead of a black box, and the **camera as the teacher** of the law. |
| Audio-visual sound-source localisation (learns *appearance* ↔ sound) | Appearance cannot separate **identical bowls**. We identify the struck bowl through **physics** (water level → predicted pitch) fused with striker motion. |
| Jal tarang apps are virtual instruments (tap a screen) | Nothing we found tunes a **real** jal tarang with a camera, or transcribes a real performance using both modalities. |

**Our three contributions**
1. **Physics-grounded audio-visual correspondence for visually identical sources.** We tell which of N identical glasses was struck by predicting each glass's pitch from its visible water level, then fuse that with striker motion using reliability-weighted Bayesian cue combination.
2. **Cross-modal self-calibration ("audio teaches vision").** Every tap gives one (fill from the camera, pitch from the mic) pair. Bayesian regression learns French's law online, and Bayesian model evidence recovers the exponent n ≈ 4 and the glass-base offset from the data. After about 5 taps, the camera **alone** predicts the pitch of any bowl at any level.
3. **Vision-guided tuning and transcription for an Indian instrument.** The system picks Sa automatically for a chosen raga, tells you *"pour X ml"* or *"STOP"* from the camera, verifies with a tap, and writes the sargam (flagging notes outside the raga) while someone plays.

---

## 2. How each modality is used

| Quantity | Camera (image) | Microphone (sound) |
|---|---|---|
| Which objects exist, where, how big | ✅ segmentation, contours, ArUco scale | ❌ |
| Water level (fill ratio h/H) | ✅ dyed-water saturation / edge profile | ❌ (only indirectly via pitch) |
| Pitch, modes, damping | ❌ | ✅ FFT peaks, STFT decay, LPC poles |
| Which bowl was struck | ✅ striker motion (frame differences) | ✅ pitch matched to physics predictions |
| Tuning target reached? | ✅ predicted from fill (continuous, while pouring) | ✅ verified by a tap (discrete) |
| The physical law linking them | supplies x = fill | supplies y = 1/f² → regression teaches the camera |

---

## 3. Model

### 3.1 Audio: the struck bowl as a source–filter system (SAP Modules 3–5)
- Source = the 2 ms contact transient (skipped); filter = the bowl. Its impulse response is a sum of damped sinusoids, i.e. an **all-pole system**.
- **Estimator 1:** zero-padded FFT (Hann window). A peak must exceed the local noise floor by 18 dB; parabolic interpolation refines it; ripples within 60 cents are merged. Per-mode decay σ comes from the slope of the STFT band energy.
- **Estimator 2:** **covariance-method LPC** (forward-only least squares, implemented from scratch). Each pole z = r·e^{jω} gives f = ω·fs/2π and σ = −fs·ln r. Burg's method was tried first and rejected: its forward–backward averaging treats a decaying ring as if it were growing when run backwards, so its poles land almost on the unit circle and the damping comes out about 99% too low, even without noise (an instructive finding for the report).
- Pitch is the lowest strong mode. Resynthesis from (f, σ, amplitude) per mode lets you hear the model.

### 3.2 Vision: classical image processing (VIP / image lectures)
- Background estimated by morphological closing on a ¼-size image; objects found as darker-than-background OR saturated pixels (Otsu threshold); then close, fill holes, and open.
- Per bowl: contour, bounding box, row-wise silhouette profile (→ volume as a solid of revolution), height and width.
- Water line: per-row median saturation in the central half of the bowl (dyed water), with an intensity-gradient fallback for clear water.
- ArUco marker → pixels per cm (for cm and ml only; the physics needs only the scale-free fill ratio).
- Tracker: stable bowl ids, plus a temporal median of the level just **before** the strike (avoids hand occlusion).

### 3.3 Physics (A. P. French, *In vino veritas*, Am. J. Phys. 1983)
f(h) = f₀ / √(1 + β·uⁿ), with u = (r − r₀)/(1 − r₀), where r is the camera fill ratio and r₀ is the base offset.
⇒ 1/f² = a + b·uⁿ is **linear** in uⁿ. This gives online Bayesian linear regression with closed-form evidence; (n, r₀) are chosen by maximum evidence with a prior n ~ N(4, 1).
The prediction uncertainty combines parameter uncertainty with vision noise propagated through the physics.

### 3.4 Fusion
- Audio likelihood: Gaussian in cents between the heard pitch and each bowl's camera-predicted pitch (with its uncertainty).
- Motion likelihood: locate the **striker tip** (the lowest pixel of what is new compared with a frame before the strike), then score each bowl with a Gaussian on the tip's distance to that bowl. A first version that summed motion energy per bowl was wrong by one bowl, because the stick's body sweeps over the neighbour; the tip-based cue fixed it.
- **Log-linear pooling with reliability weights** (audio weight from SNR, motion weight from motion contrast). When one cue degrades, the other dominates: the principle of inverse effectiveness from the course.
- A **"none of the visible bowls"** hypothesis acts as the consistency check: it rejects sounds that fit no bowl.

### 3.5 Tuning and transcription
- Range of each bowl = [pitch at 90 % full, empty pitch]. **Auto-Sa** searches for the Sa that puts every raga note inside its bowl's range with the largest safety margin.
- Target fill = inverse physics; **ml to pour** comes from the bowl's volume profile.
- **Two-step tuning (how musicians actually tune):** the camera first stops the pour one semitone (100 cents) short of the note (water can only be added easily). One verification tap measures *that bowl's* pitch offset from the shared law, then the camera guides the final top-up.
- Transcription: identified bowl → heard pitch → nearest swara relative to Sa (just intonation); notes outside the raga are shown in red.

---

## 4. Experiments (all implemented in `scripts/evaluate.py`; each result is labelled SIMULATED or REAL)
| ID | Question | Data |
|---|---|---|
| E1 | Are modal frequencies and decays measured correctly? Do FFT and LPC agree? Does resynthesis match? | simulated ground truth; ObjectFolder Real recordings (optional download) |
| E2 | Scaling law f ∝ 1/L on real size families | ObjectFolder Real + meshes (optional download) |
| E3 | Does French's law hold, and which exponent does the data choose? | simulated + **your recorded session** |
| E4 | Vision water-level accuracy under blur, darkness, noise, clear water | simulated (+ ruler check on real data) |
| E5 | Audio teaches vision: camera-only pitch error vs number of taps | simulated + your session |
| E6 | Which identical bowl was struck: chance, motion only, audio-physics only, fused, across SNR, striker visible vs hidden | simulated (live in the demo) |
| E7 | Closed-loop tune-by-sight accuracy for a raga | simulated + live demo |

## 5. Live demo (`python app/app.py`, or `--sim` without hardware)
1. **Calibrate:** strike a few bowls at different levels. The *physics plot* fills in, the fitted line appears, and n ≈ 4 emerges.
2. **Which bowl?** A volunteer strikes any of 5 identical glasses; the bowl is highlighted, and three bar sets (audio / motion / fused) show the reasoning. Hide the stick behind a hand and the audio cue carries it. Make noise and the motion cue carries it.
3. **Hear before strike:** select a bowl and press *H* to hear its predicted sound from the camera alone, then strike it to compare.
4. **Tune by sight:** choose a raga (e.g. Bhupali) and press *Plan*. Each bowl shows its target swara and "pour X ml" or "STOP"; pour and watch the gauge; strike to verify (± cents).
5. **Play & transcribe:** play a phrase; the sargam appears live, with out-of-raga notes in red. *Space* replays the last strike resynthesised from its modes.

## 6. Materials for the real demo
5–6 identical glass tumblers or bowls · food colour or haldi for the water · white sheet + lamp as a backlight · printed ArUco marker (`results/marker_4cm.png`, printed so the black square is 4 cm) · a chopstick or pencil as the striker · laptop webcam (or phone as webcam) and the laptop mic.

## 7. Implementation (complete)
```
tunesight/  config · audio · vision · physics · fusion · tuning · music · simulate · session · objectfolder
scripts/     evaluate.py · make_figures.py · build_slides.py · build_web.py · record_session.py · calibrate.py ·
             download_objectfolder.py · make_marker.py
app/app.py   PySide6 + pyqtgraph live application (live or --sim)
animations/  ManimGL scenes (3b1b manim) + render.py
web/         interactive explainer page (src template, media, figures)
slides/      IITG Beamer deck
tests/       pytest suite
```
Run order for real data: `record_session.py` → `calibrate.py` → `evaluate.py --session …` → `app/app.py`.

## 8. Limitations to state honestly
- The live room adds reverberation and noise; mitigated by a close mic and the onset/noise-floor logic.
- Clear water is harder to see than dyed water (E4 quantifies it), hence the food colour.
- Results marked SIMULATED validate the pipeline against known ground truth; REAL results come from your recorded session and ObjectFolder.
