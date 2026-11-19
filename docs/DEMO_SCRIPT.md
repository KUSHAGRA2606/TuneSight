# Live demo script (about 6 minutes)

## Before the talk
- **Table:** 5 identical clear glasses in a row on a white sheet, a lamp behind or above them, and the ArUco marker (`results/marker_4cm.png`, printed so the black square is 4 cm) beside the first glass.
- **Water:** coloured with a pinch of haldi or a drop of food colour, plus a jug.
- **Striker:** a chopstick or pencil.
- **Laptop:** webcam at glass height, about 60 cm away. Run `python app/app.py` (use `--camera 1` for a phone used as a webcam).
- **Microphone:** check it is not muted (Windows: Settings → Privacy → Microphone). A level near −97 dBFS means it is muted.
- **Backup:** if anything fails, `python app/app.py --sim` runs the same app with simulated glasses, and the interactive page `web/index.html` works offline.

## 1. The problem (30 s)
"Five identical glasses. The camera cannot hear them, and the microphone cannot tell them apart. Watch what happens when the two senses share one law."

## 2. Calibrate: audio teaches vision (90 s)
1. Choose **Mode: Calibrate**. Pour different amounts into the glasses.
2. Click a glass in the video, then strike it. Repeat on 5 to 6 glasses with different levels.
3. Point at the **Learned law** plot: the taps line up on a straight line (1/f² against uⁿ) and the title shows n close to 4, French's exponent, found from the data.
4. Each glass label now shows the pitch the camera *predicts* from the water alone.

## 3. Which glass rang? (90 s)
1. Choose **Mode: Play & transcribe**.
2. Ask someone to strike any glass. The glass is highlighted and the sargam appears at the bottom.
3. Point at the three bar sets: blue is audio (pitch against each glass's predicted pitch), orange is motion (the striker's tip), green is fused.
4. Hide the stick with your hand while striking: the motion bars go flat and audio still answers.
5. Optional: give two glasses the same water. Audio splits 50/50, and motion decides.

## 4. Tune by sight (90 s)
1. Choose a raga (Bhupali) with **Auto Sa** ticked and press **Plan tuning**.
2. Each glass shows its target note and "+X ml". Pour into one glass while watching the gauge. The camera says **STOP, tap** one semitone early.
3. Strike it once. The log shows this glass's measured offset in cents.
4. Top up until **STOP**, then strike. The big text shows the error in cents (within 25 cents is in tune).

## 5. Hear it (30 s)
- **Space** replays the last strike rebuilt from its modes (about 3 numbers per mode).
- **H** plays the selected glass's sound predicted from the camera alone, before anyone strikes it.

## If something goes wrong
| Symptom | Fix |
|---|---|
| No glasses outlined | More even backlight; move the marker away from the glasses; keep the background plain |
| Water line jumps around | Add more colour to the water; avoid reflections on the glass |
| Strikes not detected | Strike closer to the microphone; check that it is not muted |
| Wrong glass identified | Calibrate with a few more taps, spread over low and high water levels |
