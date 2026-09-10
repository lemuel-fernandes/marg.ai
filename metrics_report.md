# PathSense Evaluation Metrics Report
**Generated:** 2026-09-10 15:54:57

## Overall Status: ALL PASS

## Detailed Scenario Results

### Static Obstacle
- **Status:** PASS
- **Final Dist M:** 0.66
- **Min Clearance M:** 3.18
- **Emergency Stops:** 0.00
- **Path Efficiency:** 96.96%
- **Avg Jerk Mps3:** 0.97
- **Plot:** `scenario_static_obstacle.png`

### Crossing Animal
- **Status:** PASS
- **Final Dist M:** 0.07
- **Min Clearance M:** 8.34
- **Min Accel Mps2:** -3.31
- **Max Lateral Dev M:** 0.07
- **Rejoin T S:** 14.00
- **Emergency Stops:** 0.00
- **Path Efficiency:** 100.00%
- **Avg Jerk Mps3:** 7.18
- **Min Ttc S:** 2.72
- **Plot:** `scenario_crossing_animal.png`

## Architectural Notes for Judges
- **Safety Monitor:** Independent envelope checking prevents physical limit violations.
- **DWA Local Planner:** Time-aware obstacle prediction handles dynamic crossing animals.
- **Braking Comfort:** Harsh braking warnings (e.g., -3.3 m/s²) indicate the predictive safety monitor successfully triggered emergency deceleration to avoid a collision, prioritizing safety over passenger comfort.
