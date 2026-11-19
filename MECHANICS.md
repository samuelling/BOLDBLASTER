# Game mechanics that help model estimates

Running side notes. The approach is to make a good game first, then see which models its
dynamics can fit, rather than building a test and dressing it as a game (The Palace's approach).
Each note says what limited a model in the simulations, the mechanic that would fix it, and
games that already do it well. Numbers are from round 9 at 60 minutes unless stated; see the
dashboard's Round 9 tab and `dashboard_data/round9*.csv`.

## What each model needs from a game

| Model (candidate area) | What it needs | What limited it here | Mechanic that would help |
|---|---|---|---|
| Grid code from movement (entorhinal) | Movement in every direction, at speed | Brick Ninja only runs left or right: orientation at chance | Open arenas and foraging, not corridors; any free 2D or 3D movement works (raycaster, Betamax and Animal Sanctuary all sample directions evenly) |
| Grid code from eye movements (entorhinal) | Eye movements in every direction | Weak everywhere (phase recovery .2 to .35) even though directions are balanced: the signal is small next to the noise | Smooth pursuit of targets crossing the screen in every direction (Nau et al. used a dot moving over an arena): birds, drones, tracer fire |
| Facing direction (RSC / SPL) | Full 360 degrees of heading | Brick Ninja faces left or right only | Any free-look first-person or top-down game |
| Distance to goal, path vs Euclidean (hippocampus) | Goals behind obstacles, so path and straight-line distance diverge | Only the raycaster maze separates them, and even there they correlate .97; in the open games they are identical | Goals behind walls that force detours; locked doors and keycards (Doom), shortcuts that open later (Dark Souls), several entry points to a room (Prey, Dishonored) |
| Boundary distance (subiculum / entorhinal) | Walls of known geometry, logged | Betamax's beach and Brick Ninja have no walls the harness logs | Enclosed arenas, buildings and courtyards; log wall geometry |
| Event boundaries (hippocampus) | Boundaries that are not also arousing outcomes | Animal Sanctuary's boundaries (befriending) coincide with arousal events, so the nuisance model removes them (gain recovery .09) | Room and chapter transitions, doors, checkpoints and new objectives that are not rewards or explosions |
| Drift diffusion (pre-SMA / IPS) | For behaviour: many decisions. For BOLD: a wide range of difficulty and decisions spaced out | Behaviour recovers drift and boundary at .9 or better in every game; BOLD at most about .45 (raycaster). Betamax's targets vary little in difficulty (SD .10) and Brick Ninja's decisions are 0.8 s apart | Spaced, jittered identification decisions over a wide range of difficulty: friend-or-foe silhouettes in fog, The Palace's standing figures that may be dress forms |
| Value learning (ventral striatum) | Many outcomes per minute, ideally with contingencies that change | BOLD recovery of learning rate tracks outcome rate: .68 for Brick Ninja (about 40 a minute) against .33 for the raycaster (about 9) | Frequent outcomes; weapons, routes or loot whose payoff drifts or reverses, so the learning rate matters |
| Place familiarity (hippocampus) | Many entries into places, many of them revisits after varied delays | Betamax's bot stays near its spawn, so it enters few places (21 a minute) and encoding is poorly recovered (.45) | Backtracking through known space with changes (Metroidvania loops, keycard returns), return trips after long delays |
| Path integration (entorhinal, hippocampus, RSC) | Homing without landmarks | None of the four games has it, so it could not be tested | The Palace's lantern blackout; in Betamax a night phase where a flare dies and you must get back to the bunker |
| Map updating (hippocampus, prefrontal) | Route choices and layout changes | No route choices or layout changes in any of the four games | The Palace's Architect moving doorways; Antichamber-style rooms that lead somewhere new; in Betamax the tide closing or opening routes |

## Betamax: candidate puzzle elements

Round 9 says Betamax needs more puzzle elements: it is the best of the four for vision, attention
and reward, but its open beach gives the navigation and memory models little to work with. Each
proposal below borrows from a well-known first-person game and targets one or more models.

1. **Several routes to every objective** (Prey, Dishonored: multiple entry points, verticality,
   crawl spaces). Dunes, cottages and a seawall with two or three ways to each objective, some
   blocked by fire. Targets: path vs Euclidean distance, route-choice noise.
2. **Keycards and return trips** (Doom, Metroid Prime). Carry ammunition from a supply dump back to
   the gun line, or fetch a radio codebook from a cottage. Targets: goal distance with detours,
   place familiarity on the way back, event boundaries at pickups.
3. **Shortcuts that open** (Dark Souls). Clearing a house opens a back door that links two parts
   of the beach. Targets: map updating, successor representation.
4. **A layout that changes** (The Palace's Architect, Antichamber). The tide comes in between
   waves, closing the beach route and opening the boardwalk; shelling collapses a dune path.
   Targets: map updating (update rate), stale-route errors.
5. **Friend-or-foe in the haze** (The Palace's standing figures). Silhouettes in dawn fog that may
   be Texans or your own men, at a range of distances, a few seconds apart. Targets: drift
   diffusion from BOLD, which needs exactly this spread of difficulty and spacing.
6. **The flare dies** (The Palace's lantern trials). A night assault lit by flares; when one burns
   out, get back to the bunker in the dark with no landmarks. Targets: path integration (heading
   noise, distance gain, leak).
7. **A logic lock you can bypass** (Dishonored 2's Jindosh lock). A coded radio message to decode
   for artillery support, or an alternative route around it. Targets: structure inference;
   optional, so it never blocks play.

These keep Betamax a straightforward battle (forward mode, no possession), with the puzzles
inside the battle rather than beside it.

### Built in round 10: The Diggings

Items 1, 2, 4, 5 and 6 are now in Betamax as tunnels under the parking lot and the dunes
(branch `diggings` of samuelling/betamax): a braided maze with several routes, ammunition runs
from magazine rooms to the line, cave-ins, sector blackouts lit by a lantern, and friend-or-foe
figures. Over an hour of bot play (Experiment 9 on the dashboard):

- **Worked:** path vs Euclidean goal distance (.94, untestable on the open beach), boundary
  distance (.73), path integration heading noise (behaviour .89, BOLD .43), place familiarity
  (encoding .45 to .52, forgetting .66 to .70).
- **Cost:** 55% of the hour was spent below ground. Firefight decisions fell from 38 to 14 a
  minute, V1 SF peak from .43 to .12 (dim, uniform tunnel frames), and striatal learning rate
  from .42 to .19.
- **Too rare for BOLD:** 13 cave-ins met and 64 friend-or-foe calls an hour. Behaviour recovers
  route-choice temperature (.97) and the friend-or-foe drift (.85); BOLD recovers neither the map
  update rate nor the friend-or-foe drift.

### Tuned tunnels (built and tested)

Changes: most trips are a short cut between two hatches; magazines a short way in; cave-ins stay
down and fall on corridors you've used or the one just ahead; a figure in the corridor ahead every
8-14 s below ground; the generator blacks out the whole warren just after a box is taken; stronger
lamps. Over an hour (Experiment 9, tuned):

- **Time below ground 55% to 31%**, firefight decisions 14 to 23 a minute (Betamax 38).
- **Vision mostly back:** effective c50 .67 (Betamax .68, first tunnels .57), SF peak .22 (.43,
  .12), striatal prediction-error beta .94.
- **Navigation mostly kept:** path vs Euclidean .85 (.94), walls .63, grid orientation .61 (best of
  all three). Place forgetting .80, the best yet.
- **Map updating from choices lost its ceiling** (.63, estimates up to .95), though cave-ins were met
  less often (6 an hour); the cause isn't pinned down.
- **Still short:** homing walks fell to 9 an hour because most ammunition runs were cut short (1
  delivery an hour), friend-or-foe and map updating stay unreadable from BOLD, and striatal
  learning rate stays at .23 (Betamax .42).

Lesson: an hour of one game can't feed every model. The rare puzzle events need more hours, or a
dedicated block of tunnel play beside the beach.

## General rules so far

- Behaviour and BOLD want different things. Behaviour likes many decisions; BOLD needs them
  spaced and varied. Interleave dense action with sparse, varied decisions.
- Keep the events a model needs separate from arousing outcomes, or the nuisance model removes them.
- Free 2D or 3D movement is enough for direction codes; a side-scroller is not.
- Medial temporal signals are small and noisy (noise 1.4 to 1.6 times cortex here), so those
  models need the most minutes of the right events.
- Measuring gaze well matters most for V1; large receptive fields and navigation codes are far
  more tolerant (round 6).
- Puzzle time is borrowed from action time. Every minute in a maze is a minute without firefights,
  outcomes and varied images, so puzzle elements should be short and dense, not long errands
  (round 10).
- Building the game around the models works for navigation and memory: walls everywhere, rooms
  revisited on three floors, and detours to every goal gave the best boundary, grid, event and
  familiarity recovery yet (round 11, Halcyon).
- Count decisions per encounter, not per person. A group of five people is one decision for BOLD;
  treating each as a call put them 2.8 s apart and erased drift diffusion and value learning from
  BOLD (round 11).
- Indoor scenes need a wide contrast range and some daylight; a dim hotel cost effective c50.

### Round 12 addendum

- Spacing decisions out is not enough by itself: going from 2.8 s to 9.4 s between calls halved the
  calls per hour and BOLD-based decision parameters got worse. Trial count and spacing have to be
  tuned together; about 6 s is the next thing to try.
- A hidden, slowly drifting risk with a clear outcome at a fixed place (the stair landing) is
  recoverable from striatal BOLD once there are about 100 trials an hour (r 0.43).

### Round 13 addendum

- One hour is a noisy ruler for BOLD-based recovery of sparse events: the same game on another hour
  moves r by about 0.15. Compare designs on replicate hours, not single ones.
- Spacing decisions out does not help their recovery from BOLD; the number of decisions and the
  number of hours do. Dense calls and several hours per person.
- Haze (volumetric fog, bloom) costs about a quarter of the frame contrast in every band and some of
  the spatial-frequency fit. A view model (arms, weapon) costs nothing measurable.
