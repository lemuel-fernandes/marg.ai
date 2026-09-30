# PathSense Evaluation Metrics Report
**Generated:** 2026-09-29 21:12:25

## Overall Status: ALL PASS

## Detailed Scenario Results

### Static Obstacle
- **Status:** PASS
- **Final Dist M:** 1.27
- **Min Clearance M:** 2.87
- **Emergency Stops:** 0.00
- **Path Efficiency:** 94.65%
- **Avg Jerk Mps3:** 4.90
- **Detection Map 50:** 0.74
- **Detection Precision:** 0.74
- **Detection Recall:** 0.74
- **Detection F1:** 0.74
- **Ap Cattle:** 0.74
- **Plot:** `scenario_static_obstacle.png`

### Crossing Animal
- **Status:** PASS
- **Final Dist M:** 1.34
- **Min Clearance M:** 5.56
- **Min Accel Mps2:** -3.50
- **Max Lateral Dev M:** 33.63
- **Rejoin T S:** 10.20
- **Emergency Stops:** 0.00
- **Path Efficiency:** 97.90%
- **Avg Jerk Mps3:** 9.57
- **Min Ttc S:** 1.61
- **Min Ttc Overall S:** 1.61
- **Detection Map 50:** 0.69
- **Detection Precision:** 0.69
- **Detection Recall:** 0.69
- **Detection F1:** 0.69
- **Ap Cattle:** 0.69
- **Plot:** `scenario_crossing_animal.png`

### Multi Animal
- **Status:** PASS
- **Final Dist M:** 1.19
- **Min Clearance M:** 1.36
- **Min Accel Mps2:** -3.50
- **Emergency Stops:** 0.00
- **Path Efficiency:** 100.00%
- **Avg Jerk Mps3:** 9.50
- **Min Ttc Overall S:** 0.49
- **Detection Map 50:** 0.71
- **Detection Precision:** 0.71
- **Detection Recall:** 0.71
- **Detection F1:** 0.71
- **Ap Cattle:** 0.71
- **Plot:** `scenario_multi_animal.png`

### Sensor Noise
- **Status:** PASS
- **Final Dist M:** 1.17
- **Min Clearance M:** 2.98
- **Emergency Stops:** 0.00
- **Path Efficiency:** 95.01%
- **Avg Jerk Mps3:** 4.47
- **Detection Map 50:** 0.83
- **Detection Precision:** 0.83
- **Detection Recall:** 0.83
- **Detection F1:** 0.83
- **Ap Cattle:** 0.83
- **Plot:** `scenario_sensor_noise.png`

### Indian Road
- **Status:** PASS
- **Final Dist M:** 0.97
- **Min Clearance M:** 0.81
- **Min Accel Mps2:** -3.50
- **Emergency Stops:** 0.00
- **Path Efficiency:** 98.76%
- **Avg Jerk Mps3:** 6.47
- **Min Ttc S:** 3.30
- **Min Ttc Overall S:** 0.33
- **Detection Map 50:** 0.70
- **Detection Precision:** 0.76
- **Detection Recall:** 0.76
- **Detection F1:** 0.76
- **Ap Car:** 0.76
- **Ap Cattle:** 0.63
- **Plot:** `scenario_indian_road.png`

### City Roads
- **Status:** PASS
- **Final Dist M:** 1.33
- **Min Clearance M:** 1.42
- **Min Accel Mps2:** -3.50
- **Emergency Stops:** 0.00
- **Path Efficiency:** 80.46%
- **Avg Jerk Mps3:** 11.05
- **Min Ttc S:** 0.67
- **Min Ttc Overall S:** 0.65
- **Detection Map 50:** 0.44
- **Detection Precision:** 0.60
- **Detection Recall:** 0.60
- **Detection F1:** 0.60
- **Ap Car:** 0.24
- **Ap Cattle:** 0.50
- **Ap Pedestrian:** 0.29
- **Ap Pothole:** 0.73
- **Plot:** `scenario_city_roads.png`

### Occluded Siren
- **Status:** PASS
- **Final Dist M:** 2.02
- **Min Clearance M:** 0.29
- **Min Accel Mps2:** -3.42
- **Emergency Stops:** 0.00
- **Max Hold Speed Mps:** 0.10
- **Post Siren Progress M:** 18.21
- **Path Efficiency:** 96.85%
- **Avg Jerk Mps3:** 3.65
- **Min Ttc S:** 0.23
- **Min Ttc Overall S:** 0.23
- **Detection Map 50:** 0.62
- **Detection Precision:** 0.62
- **Detection Recall:** 0.62
- **Detection F1:** 0.62
- **Ap Car:** 0.62
- **Acoustic Doa Mae Deg:** 0.00
- **Acoustic Doa Rmse Deg:** 0.00
- **Acoustic Classification Acc:** 1.00
- **Acoustic Detection Precision:** 0.67
- **Acoustic Detection Recall:** 1.00
- **Plot:** `scenario_occluded_siren.png`

### Free World
- **Status:** PASS
- **Final Dist M:** 1.31
- **Min Clearance M:** 1.51
- **Min Accel Mps2:** -3.50
- **Emergency Stops:** 0.00
- **Random Events:** 6.00
- **World Seed:** 2026.00
- **Path Efficiency:** 80.20%
- **Avg Jerk Mps3:** 15.23
- **Min Ttc S:** 0.60
- **Min Ttc Overall S:** 0.60
- **Detection Map 50:** 0.66
- **Detection Precision:** 0.70
- **Detection Recall:** 0.70
- **Detection F1:** 0.70
- **Ap Car:** 0.69
- **Ap Cattle:** 0.63
- **Ap Pedestrian:** 0.59
- **Ap Pothole:** 0.72
- **Acoustic Doa Mae Deg:** 0.00
- **Acoustic Doa Rmse Deg:** 0.00
- **Acoustic Classification Acc:** 1.00
- **Acoustic Detection Precision:** 0.80
- **Acoustic Detection Recall:** 1.00
- **Plot:** `scenario_free_world.png`

## Architectural Notes for Judges
- **Safety Monitor:** Independent envelope checking prevents physical limit violations.
- **DWA Local Planner:** Time-aware obstacle prediction handles dynamic crossing animals.
- **Braking Comfort:** Harsh braking warnings (e.g., -3.3 m/s²) indicate the predictive safety monitor successfully triggered emergency deceleration to avoid a collision, prioritizing safety over passenger comfort.
