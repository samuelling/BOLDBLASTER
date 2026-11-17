# Halcyon: design

Halcyon is a first-person shooter built backwards from BOLDBLASTER's results. Rounds 8-10 asked
how well each brain model can be recovered from an hour of someone playing a game. The lesson of
the Betamax tunnels was that **time spent on one model is borrowed from the others**: tunnels
bought navigation and lost vision and reward. So Halcyon has no separate puzzle area. Every
mechanic runs inside the same moment-to-moment fight, so each minute of play feeds every model.

## Setting

The Halcyon is a derelict 1960s seaside hotel on a cliff. Gunmen hold the guests. You clear it
floor by floor, find guests and walk them out. The building is the level: three floors around a
central atrium, a grand stair and two service stairwells, guest corridors in a loop, a ballroom,
a kitchen, a pool terrace open to the sky, and the sea beyond the windows.

## What each model needs, and the mechanic that supplies it

| Model (candidate area) | What recovery needed in rounds 8-10 | Mechanic in Halcyon |
|---|---|---|
| V1 tuning: SF, contrast, surround (V1) | Varied, textured images; time in bright, high-contrast scenes; gaze where the model thinks it is. Dim, uniform tunnels lost most of the SF signal. | Scanned PBR materials (patterned carpet, tile, wallpaper, wood, brass), every room lit differently (sun through windows, lamps, neon, flicker), outdoor terrace. Fights happen at the crosshair: targets appear near eye height in front of you, so gaze sits at screen centre more of the time. |
| Numerosity (parietal maps) | A varied number of items in view. | Groups of 1-5 people per encounter. |
| Self-motion (hMT+/MST) | Varied forward motion and turning. | Corridors, stairs, strafing in rooms, turns at every junction. |
| Striatal RPE / value learning (ventral striatum) | Outcomes that differ from expectation, spaced far enough apart for BOLD. Betamax's 38 bursts a minute were too dense; the tunnels' outcomes were too few. | **Stairwell choice as a restless bandit**: every floor change is a choice of grand stair, east or west stairwell. Each has a hidden chance of an ambush that drifts slowly; a clear stairwell often has supplies. About 2 choices a minute, each with a clear outcome a few seconds later. |
| Drift diffusion (pre-SMA / IPS) | Many decisions with graded evidence, 5-15 s apart. Bursts were too dense; friend-or-foe figures were too few. | **Shoot or hold**: every encounter is a call between gunman and guest (armed and raised vs hands up or phone, uniform vs hotel clothes) at varied distance, light and partial cover. One every 4-10 s while moving. |
| Goal distance, path vs Euclidean (hippocampus) | Goals whose route differs from the straight line. | Objectives on other floors and round corners: a guest in room 214, then out to the terrace. Floors make path and straight-line distance diverge everywhere. |
| Boundary distance (subiculum) | Walls at varied distances. | Interior rooms and corridors throughout; distance to the nearest wall is logged every frame. |
| Facing direction, grid code (RSC/SPL, EC) | Movement in every direction across open 2D space. | The atrium floor, ballroom and pool terrace are open; corridors run in four directions. |
| Event boundaries (hippocampus) | Clear transitions. | Room-to-room transitions, floor changes, objective pick-ups and hand-offs, contacts. |
| Place familiarity (hippocampus) | Many entries, many revisits at varied gaps. | A compact building visited repeatedly in different orders. |
| Map updating (hippocampus) | Changes to the layout that are met, and met again. The tunnels' cave-ins were met 6-13 times an hour. | **Security doors**: the hotel's fire doors open and lock on their own (the gang runs the system). Doors are on the corridor loop and the stairwells you use, so a changed door is met within a minute and often again. Target 60+ an hour. |
| Path integration (EC/hippocampus) | Walks in the dark from a known start back to a known place. The tunnels produced 9-18 an hour. | **Blackouts on the walk out**: after you pick up a guest the power often fails; you lead them back to the stair you came up by torchlight. Target 25+ an hour. |

## Rates to hit (per hour of play)

Shoot-or-hold calls 300+, stairwell choices 100+, door changes met 60+, homing walks 25+, objectives
30+, time in darkness under 15%, time outdoors 10-20%.

## Constraints

- Forward play only; a straightforward fight. No possession or rewind.
- One continuous loop, no menus mid-play: an endless stream of objectives.
- Everything the models need is logged every frame (position, floor, facing, visible people with
  screen position and role, distance to walls, light level, room) and as events (calls, choices,
  outcomes, door changes met, blackouts, objectives).
- Assets: CC0 (Poly Haven, ambientCG) where possible. Characters are the exception (see README).
