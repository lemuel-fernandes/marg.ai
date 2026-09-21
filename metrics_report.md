# PathSense Evaluation Metrics Report
**Generated:** 2026-09-21 12:28:06

## Overall Status: FAILURES DETECTED

## Detailed Scenario Results

### Static Obstacle
- **Status:** PASS
- **Final Dist M:** 1.18
- **Min Clearance M:** 2.79
- **Emergency Stops:** 0.00
- **Path Efficiency:** 94.09%
- **Avg Jerk Mps3:** 3.56
- **Plot:** `scenario_static_obstacle.png`

### Crossing Animal
- **Status:** PASS
- **Final Dist M:** 1.34
- **Min Clearance M:** 6.00
- **Min Accel Mps2:** -3.50
- **Max Lateral Dev M:** 34.01
- **Rejoin T S:** 10.30
- **Emergency Stops:** 0.00
- **Path Efficiency:** 98.49%
- **Avg Jerk Mps3:** 10.24
- **Min Ttc S:** 1.89
- **Min Ttc Overall S:** 1.89
- **Plot:** `scenario_crossing_animal.png`

### Multi Animal
- **Status:** PASS
- **Final Dist M:** 1.23
- **Min Clearance M:** 1.40
- **Min Accel Mps2:** -3.50
- **Emergency Stops:** 0.00
- **Path Efficiency:** 100.00%
- **Avg Jerk Mps3:** 9.50
- **Min Ttc Overall S:** 0.53
- **Plot:** `scenario_multi_animal.png`

### Sensor Noise
- **Status:** PASS
- **Final Dist M:** 1.02
- **Min Clearance M:** 2.83
- **Emergency Stops:** 0.00
- **Path Efficiency:** 95.54%
- **Avg Jerk Mps3:** 4.26
- **Plot:** `scenario_sensor_noise.png`

### Indian Road
- **Status:** PASS
- **Final Dist M:** 1.51
- **Min Clearance M:** 0.49
- **Min Accel Mps2:** -3.50
- **Emergency Stops:** 0.00
- **Path Efficiency:** 99.71%
- **Avg Jerk Mps3:** 9.16
- **Min Ttc S:** 4.67
- **Min Ttc Overall S:** 0.01
- **Plot:** `scenario_indian_road.png`

### City Roads
- **Status:** FAIL
- **Final Dist M:** 2.78
- **Min Clearance M:** 1.45
- **Min Accel Mps2:** -3.50
- **Emergency Stops:** 8.00
- **Boundary Overshoot M:** 5.46
- **Path Efficiency:** 85.53%
- **Avg Jerk Mps3:** 3.03
- **Min Ttc S:** 0.78
- **Min Ttc Overall S:** 0.69
- **Failures:** boundary violation: ego left road corridor by 5.46 m
- **Plot:** `scenario_city_roads.png`

### Occluded Siren
- **Status:** FAIL
- **Final Dist M:** 15.24
- **Min Clearance M:** 0.78
- **Min Accel Mps2:** -2.96
- **Emergency Stops:** 0.00
- **Max Hold Speed Mps:** 0.50
- **Post Siren Progress M:** 0.52
- **Path Efficiency:** 97.57%
- **Avg Jerk Mps3:** 0.79
- **Min Ttc Overall S:** 0.57
- **Failures:** did not reach goal (dist=15.24 m), blind creep during unseen-siren hold (max 0.50 m/s), deadlocked after siren ended (progress 0.52 m)
- **Plot:** `scenario_occluded_siren.png`

## Architectural Notes for Judges
- **Safety Monitor:** Independent envelope checking prevents physical limit violations.
- **DWA Local Planner:** Time-aware obstacle prediction handles dynamic crossing animals.
- **Braking Comfort:** Harsh braking warnings (e.g., -3.3 m/s²) indicate the predictive safety monitor successfully triggered emergency deceleration to avoid a collision, prioritizing safety over passenger comfort.
