  # PathSense — Production Roadmap & Sprint Plan
### SIH26037 · Adaptive Path Planning & Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

**Today:** Thu, 10 Sep 2026 · **Internal deadline:** Fri, 11 Sep 2026 · **Official PS reference date:** 30 Sep 2026

> This assumes the 11 Sep deadline is your **idea/PPT + team-plan submission** (internal or SIH portal round), not the working prototype. If 11 Sep is actually a working-demo deadline, skip straight to the "🚨 If demo is due tomorrow" section at the bottom.

---

## 1. Team & Roles (6 members)

| # | Role | Owns | Primary Skills Needed |
|---|------|------|------------------------|
| **M1** | Team Lead / System Integration | Architecture decisions, sprint tracking, module integration, final demo assembly | Systems thinking, Git, project management |
| **M2** | Perception Engineer | Obstacle/pothole/animal/pedestrian detection from camera or simulated sensor data | Computer vision, OpenCV/PyTorch |
| **M3** | Global Path Planning Engineer | Route-level planning: A*/Hybrid A*/RRT* on the costmap | Algorithms, graph search, MATLAB or Python |
| **M4** | Local Planning & Collision Avoidance Engineer | Real-time re-planning + collision avoidance (DWA/MPC), vehicle dynamics constraints | Control theory, optimization |
| **M5** | Simulation & Test Engineer | Simulation environment (MATLAB/Simulink or ROS2+Gazebo/CARLA), Indian-road test scenarios, metrics | Simulation tools, scenario design |
| **M6** | Dashboard, Docs & Presentation Lead | Live visualization dashboard, README/docs, PPT, demo video, judge Q&A prep | Frontend (React/Plotly), storytelling, video editing |

Everyone reviews everyone else's PRs before merge — 2-person minimum review on anything touching the planner or controller (safety-critical path).

---

## 2. Sprint Overview

| Sprint | Dates | Goal |
|--------|-------|------|
| **Sprint 0 — Emergency** | Today (10 Sep, tonight) → 11 Sep AM | Idea locked, architecture decided, repo + roles set, PPT submitted |
| **Sprint 1 — Foundations** | 12–16 Sep | Simulation environment running, dummy perception + dummy planner talking to each other end-to-end (even if crude) |
| **Sprint 2 — Core Algorithms** | 17–22 Sep | Real perception module, real global planner, first version of local collision avoidance |
| **Sprint 3 — Integration & Hardening** | 23–27 Sep | Full pipeline integrated, dashboard live, test-scenario suite run, metrics collected, bugs fixed |
| **Sprint 4 — Polish & Submission** | 28–30 Sep | Demo video, final PPT, docs, rehearsal, buffer for last-minute fixes |

**Golden rule:** at the end of every sprint there must be a runnable, demoable version — even if ugly. Never let integration wait until the last sprint.

---

## 3. Sprint 0 — Emergency (due 11 Sep AM) 🚨

**Goal:** nothing coded yet needs to be perfect — but the *plan* and the *story* must be bulletproof, because that's what's being judged tomorrow.

### Issues

- **#1 [M1] Lock the technical approach** — *Est: 1 hr*
  - Decide: MATLAB/Simulink-first (safer, matches sponsor "MathWorks") vs ROS2+Gazebo/CARLA-first (more flexible, more familiar to most teams).
  - Acceptance: one paragraph in README stating the chosen stack + why.

- **#2 [M1] Create GitHub repo + board** — *Est: 30 min*
  - Repo with `README.md`, `ROADMAP.md`, `/docs`, `/src`, `/sim`, `/dashboard` folders.
  - GitHub Projects (or Trello/Notion) board with all issues below as cards, columns: Backlog / In Progress / Review / Done.
  - Acceptance: every teammate has push access and sees their assigned issues.

- **#3 [M2] List obstacle categories for Indian unstructured roads** — *Est: 1 hr*
  - Cows/stray animals, pedestrians crossing anywhere, two-wheelers weaving, potholes, encroached carts/stalls, oncoming wrong-side vehicles, unmarked speed breakers.
  - Acceptance: a table in `/docs/scenarios.md` (category → why it's hard → how we'll simulate it).

- **#4 [M3 + M4] Sketch the planning pipeline diagram** — *Est: 1 hr*
  - One diagram: sensors → perception → costmap → global planner → local planner/collision avoidance → controller → vehicle/sim.
  - Acceptance: diagram embedded in README (already drafted — refine it together).

- **#5 [M5] Shortlist simulation scenarios for the demo** — *Est: 1 hr*
  - 3–5 concrete scenarios you will actually show judges, e.g.: "goat crosses road suddenly," "pothole cluster + oncoming auto," "narrow lane with parked cart."
  - Acceptance: listed in `/docs/scenarios.md` with expected before/after behaviour.

- **#6 [M6] Build the idea PPT** — *Est: 2–3 hrs*
  - Slides: Problem → Why unstructured Indian roads are different → Proposed approach/architecture → Tech stack → Innovation/USP → Feasibility → Impact → Team.
  - Acceptance: PPT reviewed by all 6, uploaded to portal/shared drive before 11 Sep deadline.

- **#7 [ALL] 15-min sync tonight** — confirm roles, confirm Sprint 1 issues below, raise blockers.

---

## 4. Sprint 1 — Foundations (12–16 Sep)

**Goal:** a thin, ugly, end-to-end pipeline running in simulation — proves the architecture works before anyone polishes anything.

- **#8 [M5] Stand up simulation environment** — *Est: 2 days*
  - Install & configure chosen sim (MATLAB Automated Driving Toolbox / Simulink 3D Animation, or ROS2+Gazebo/CARLA).
  - Acceptance: one scripted scenario (straight road + one static obstacle) runs and can be visualized.

- **#9 [M2] Dummy/simple obstacle detector** — *Est: 2 days*
  - Even a basic color/blob or ground-truth-from-simulator obstacle list is fine for now.
  - Acceptance: outputs a list of obstacle positions per simulation timestep, published to a shared interface (topic/struct/JSON — agree format with M1).

- **#10 [M3] Static global planner (A*/Hybrid A*)** — *Est: 3 days*
  - Given a start, goal, and static occupancy grid, output a feasible path.
  - Acceptance: path visualized on the sim map for at least 2 test layouts.

- **#11 [M4] Vehicle kinematic model + naive controller** — *Est: 2 days*
  - Bicycle model (or simulator's built-in vehicle model) + pure pursuit or simple PID path follower.
  - Acceptance: vehicle follows M3's static path in sim without crashing into a static obstacle.

- **#12 [M6] Wireframe the dashboard** — *Est: 1 day*
  - Decide what judges need to see live: map + obstacles + planned path + vehicle state + safety margin.
  - Acceptance: clickable wireframe or basic React shell with placeholder data.

- **#13 [M1] Define shared data contracts** — *Est: 1 day (do this FIRST, before #8–#12 diverge)*
  - Agree exact message formats between perception → planner → controller → dashboard (field names, units, coordinate frame).
  - Acceptance: `/docs/interfaces.md` that everyone codes against.

**Sprint 1 exit demo:** vehicle drives a straight-ish path around one static obstacle in sim, visible on a basic dashboard.

---

## 5. Sprint 2 — Core Algorithms (17–22 Sep)

- **#14 [M2] Real perception module** — *Est: 3 days*
  - CV/DL model (or simulated LiDAR + clustering) to detect and classify: static obstacles, pedestrians/animals (dynamic), potholes.
  - Acceptance: correctly flags dynamic vs static obstacle type in ≥80% of test-scenario frames.

- **#15 [M3] Dynamic re-planning on the global planner** — *Est: 2 days*
  - Global path recalculates when the costmap changes materially (new obstacle appears blocking route).
  - Acceptance: demonstrated on a scenario where a "road" gets fully blocked and the planner reroutes.

- **#16 [M4] Local planner / collision avoidance (DWA or MPC)** — *Est: 4 days* — the core innovation of this PS, give it the most time.
  - Reacts to sudden dynamic obstacles (animal/pedestrian crossing) within the vehicle's dynamic limits (max steering angle, braking distance).
  - Acceptance: vehicle avoids a suddenly-appearing dynamic obstacle in ≥3 different scenarios without exceeding defined jerk/deceleration limits.

- **#17 [M4 + M3] Global ↔ Local planner handshake** — *Est: 1 day*
  - Local planner deviates from global path to avoid a collision, then rejoins it.
  - Acceptance: visible in sim + logged path deviation stays within a defined safe corridor.

- **#18 [M5] Build the full test-scenario suite** — *Est: 2 days*
  - Script all scenarios from `/docs/scenarios.md` as repeatable, one-command-run sim scripts.
  - Acceptance: `run_all_scenarios.sh` (or MATLAB equivalent) runs every scenario headlessly and logs results.

- **#19 [M5] Define & compute evaluation metrics** — *Est: 1 day*
  - Collision count, minimum time-to-collision, path length vs optimal, average jerk/acceleration, planning latency.
  - Acceptance: a results table auto-generated after a test run.

- **#20 [M6] Wire dashboard to real data** — *Est: 2 days*
  - Replace placeholder data with live feed from perception/planner/controller.
  - Acceptance: dashboard shows real-time obstacle detections + planned path + vehicle trail during a live sim run.

**Sprint 2 exit demo:** vehicle handles a sudden dynamic obstacle (e.g., animal crossing) live, dashboard reflects it in real time, metrics are logged.

---

## 6. Sprint 3 — Integration & Hardening (23–27 Sep)

- **#21 [M1] Full pipeline integration pass** — *Est: 2 days*
  - Run perception → costmap → global planner → local planner → controller → dashboard as one continuous system, no manual glue.
  - Acceptance: single command/script launches the entire pipeline end-to-end.

- **#22 [ALL] Run full scenario suite, triage bugs** — *Est: 2 days*
  - Every member takes ownership of bugs in their own module; cross-review for integration bugs.
  - Acceptance: all scenarios in `/docs/scenarios.md` pass without collision (or failures are documented + explained).

- **#23 [M2] Improve perception edge cases** — *Est: 2 days*
  - Occlusion, small/low-contrast obstacles (potholes), overlapping dynamic obstacles.
  - Acceptance: false-negative rate documented and reduced from Sprint 2 baseline.

- **#24 [M4] Tune collision-avoidance safety margins** — *Est: 1–2 days*
  - Balance "too cautious/stops constantly" vs "cuts it too close."
  - Acceptance: metrics from #19 show zero collisions across the full scenario suite at acceptable jerk/comfort levels.

- **#25 [M5] Stress-test scenarios** — *Est: 1 day*
  - Multiple simultaneous dynamic obstacles, sensor noise injection, degraded visibility.
  - Acceptance: documented pass/fail + known limitations list (judges respect honesty about limitations).

- **#26 [M6] Finish dashboard polish + screen-recording setup** — *Est: 1–2 days*
  - Clean UI, legends, safety-margin indicator, replay/scrub capability if time allows.
  - Acceptance: dashboard is demo-ready without a developer narrating what each element means.

- **#27 [M1 + M6] (Stretch) Small-scale hardware demo** — *Est: whatever's left*
  - If a robot-car rig is feasible, port the local planner + controller to it for a physical obstacle-course demo.
  - Acceptance: only attempt if Sprints 1–3 core issues are all closed — this is a bonus, not a requirement.

- **#27b [M2 + M4] (Stretch) Audio / siren-detection module ("acoustic attention")** — *Est: 1–2 days*
  - **Motivation:** on unstructured Indian roads, sound often arrives before sight — horns from wrong-side/oncoming traffic around blind bends, ambulance sirens occluded by trucks, procession loudspeakers. A small microphone array lets the vehicle "hear around" occlusions that the camera optical fallback (`VisionPerceptionPipeline` track persistence + occlusion speed cap) cannot. This complements vision; it never replaces it.
  - **Design sketch (new `src/perception/audio_pipeline.py`):**
    1. *Capture* — 2–4 mic array @ 16 kHz; high-pass + AGC frontend.
    2. *Features* — 25 ms frames → log-mel spectrogram; energy focus in horn/siren bands (300 Hz–4 kHz).
    3. *Detection* — lightweight classifier over event windows: {horn, siren, rickshaw_hooter, background} (1D-CNN/MobileNet-grade on hardware; band-energy + spectral-flux heuristics in simulation). Emits class, confidence, onset/offset.
    4. *DOA* — GCC-PHAT TDOA across mic pairs → azimuth in base_link; ±15° resolution is sufficient to bias attention.
    5. *Fusion contract* — implement the existing `PerceptionNode` contract as `AcousticPerceptionNode`; publish `AttentionCue {azimuth, class, confidence, t_onset}` on a new `Topic.ACOUSTIC_CUE`.
    6. *Planner coupling* — cues (a) extend vision track persistence for tracks inside the cue azimuth cone, and (b) drop the occlusion-horizon speed cap (creep) when a siren/horn cue has no matching visual track within ~2 s — i.e., something is coming that we cannot see yet. Wire into `BehaviorStateMachine` CREEP_AND_YIELD preconditions.
    7. *Simulation without real audio* — scenario-level synthetic event injection: a scenario declares `acoustic_events = [(t_onset, class, azimuth, snr_db)]` and the audio node consumes these in place of real captures, so the full fusion path stays deterministic and testable.
  - **Metrics (extends #19):** per-class detection precision/recall, false alarms per minute, cue latency (onset → bus publish), end-to-end reaction (siren onset → speed-cap drop ≤ 1.0 s at the 10 Hz sim tick).
  - Acceptance: with a synthetic siren spawned from an occluded approach in one scenario, the occlusion speed-cap engages from the acoustic cue alone; precision ≥ 80% and recall ≥ 80% on the synthetic event set; zero false-positive creep episodes across the existing 6-scenario suite (no acoustic events present).

**Sprint 3 exit demo:** full pipeline runs unattended through the whole scenario suite with zero collisions and a polished live dashboard.

---

## 7. Sprint 4 — Polish & Submission (28–30 Sep)

- **#28 [M6] Record final demo video** — *Est: 1 day* — 2–4 min, problem → solution → live results → impact.
- **#29 [M6] Finalize PPT** — *Est: 1 day* — updated with real results/metrics/screenshots, not just concept art.
- **#30 [M1] Finalize README + architecture docs** — *Est: half day*
- **#31 [ALL] Full dry-run presentation + Q&A prep** — *Est: half day* — anticipate: "why not just use lane detection," "how does this generalize beyond your test scenarios," "what's your actual novelty vs published papers."
- **#32 [M1] Buffer day** — *Est: 1 day* — reserved purely for whatever breaks. Do not schedule new features here.

---

## 8. Deliverables Checklist (@all — confirm/add before dividing work further)

Add/edit anything missing, then we lock this and split remaining work:

- [ ] **Idea PPT** (Sprint 0) — problem framing, architecture, novelty, feasibility, impact
- [ ] **GitHub repo** — clean structure, README, ROADMAP, issues tracked
- [ ] **Architecture & interface docs** (`/docs/interfaces.md`, diagram)
- [ ] **Working simulation environment** with all listed unstructured-road scenarios scripted
- [ ] **Perception module** with documented accuracy on the scenario suite
- [ ] **Global path planner** with reroute-on-blockage capability
- [ ] **Local planner / collision-avoidance module** — the core differentiator, needs the most rigor and the most test coverage
- [ ] **Vehicle controller** respecting real kinematic/dynamic constraints
- [ ] **Live dashboard** showing obstacles, path, vehicle state, safety margin
- [ ] **Evaluation metrics report** (collision rate, path efficiency, comfort/jerk, planning latency) — this is what separates a "cool demo" from a "credible engineering result" in judges' eyes
- [ ] **Documented limitations** — what scenarios it does NOT yet handle, and why (shows engineering maturity)
- [ ] **Demo video**
- [ ] **Final PPT** with real results
- [ ] (Stretch) **Small-scale hardware validation**
- [ ] (Stretch) **Acoustic attention module** (#27b — siren/horn detection + occlusion-aware speed coupling)

---

## 9. 🚨 If the working demo is due tomorrow (11 Sep), not just the idea

Compress everything above into one night:
1. **[M1]** Freeze scope to ONE scenario type (e.g., one static + one dynamic obstacle) — do not attempt the full scenario suite.
2. **[M5]** Get any simulator running with a vehicle + a hardcoded obstacle, even with placeholder/mock data — speed over realism.
3. **[M3+M4]** Implement the simplest working planner (e.g., A* global + basic DWA/pure-pursuit local) — algorithmic elegance can wait for later sprints.
4. **[M2]** Use ground-truth obstacle positions from the simulator instead of real perception — real CV comes in Sprint 2.
5. **[M6]** Skip the fancy dashboard — a basic 2D plot of path + obstacles + vehicle is enough for tomorrow.
6. **[ALL]** Prioritize a clean, honest PPT that says "here's what works today, here's the roadmap for what's next" over a broken from over-scoped demo.

---

*Update the Status checklist in README.md at the end of every sprint. Keep issues in sync with the GitHub board — this file is the source of truth for "what to do," the board is the source of truth for "who's doing it right now."*
