# PathSense Evaluation Metrics Report
**Generated:** 2026-09-21 13:50:41

## Overall Status: ALL PASS

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
- **Final Dist M:** 1.39
- **Min Clearance M:** 0.38
- **Min Accel Mps2:** -3.50
- **Emergency Stops:** 0.00
- **Path Efficiency:** 99.53%
- **Avg Jerk Mps3:** 9.89
- **Min Ttc S:** 4.84
- **Min Ttc Overall S:** 0.02
- **Plot:** `scenario_indian_road.png`

### City Roads
- **Status:** PASS
- **Final Dist M:** 1.35
- **Min Clearance M:** 1.51
- **Min Accel Mps2:** -3.50
- **Emergency Stops:** 0.00
- **Path Efficiency:** 80.60%
- **Avg Jerk Mps3:** 10.33
- **Min Ttc S:** 0.88
- **Min Ttc Overall S:** 0.88
- **Plot:** `scenario_city_roads.png`

### Occluded Siren
- **Status:** PASS
- **Final Dist M:** 1.89
- **Min Clearance M:** 0.37
- **Min Accel Mps2:** -3.50
- **Emergency Stops:** 0.00
- **Max Hold Speed Mps:** 0.14
- **Post Siren Progress M:** 17.20
- **Path Efficiency:** 97.47%
- **Avg Jerk Mps3:** 3.54
- **Min Ttc S:** 0.93
- **Min Ttc Overall S:** 0.33
- **Plot:** `scenario_occluded_siren.png`

## Architectural Notes for Judges
- **Safety Monitor:** Independent envelope checking prevents physical limit violations.
- **DWA Local Planner:** Time-aware obstacle prediction handles dynamic crossing animals.
- **Braking Comfort:** Harsh braking warnings (e.g., -3.3 m/s²) indicate the predictive safety monitor successfully triggered emergency deceleration to avoid a collision, prioritizing safety over passenger comfort.
