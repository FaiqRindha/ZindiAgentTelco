PROMPTS = {
'add_neighbor' : '''
For “add neighbor” options, base the choice on the failure window only:
1. Identify the serving cell during the throughput degradation.
2. Identify the neighbor that is strongest or repeatedly better during that same interval.
3. Use A3/A5/re-establishment events to confirm the likely source→target HO path.
4. Choose the option that matches:
   source = degraded serving cell
   target = best candidate neighbor
Do not choose based only on geography, configured neighbor lists, or the strongest sample outside the degraded period.  
''',

'modify_pdcch' : '''
If multiple solutions suggest Modify PdcchOccupiedSymbolNum, select the one where:
1. The cell is the serving cell during the throughput degradation window (match serving PCI).
2. The degradation shows PDCCH symptoms (high CCE Fail Rate + low Grant + sharp throughput drop).

Reject all others:
- Neighbor cells
- Cells not serving during the issue window
- Cells without clear PDCCH bottleneck evidence
''',

'decrease_threshold' : '''
When resolving throughput drops via "Decrease Threshold" solutions, choose the best solution by:
-Source Identification: Locate the Serving PCI at the timestamp immediately before the handover/throughput drop.
-Ownership Rule: A2/A5 thresholds are Serving Cell parameters. Only the serving cell triggers the measurement process.
-Strict Selection: Match the solution to the Source Cell ID identified in Step 1.
-Neighbor Exclusion: Explicitly REJECT any solution targeting a Neighbor/Target PCI, regardless of its signal strength. The fix must be applied to the "gatekeeper" (the Serving Cell).
''',

'tilt_up' : '''
When choosing between "Lift Tilt" solutions, follow this priority logic:
1. Identify the Bottleneck: Locate the cell with the highest PRB Utilization (e.g., >70%). This cell is "Congested."
2. Apply the Load Balancing Filter:
- REJECT lifting the tilt of a Serving Cell that is already congested. (Lifting tilt expands its footprint, adding more load and worsening resource starvation).
- SELECT lifting the tilt of a Neighbor Cell that has low utilization. This allows the idle cell to "reach" the user's location and trigger a handover to offload the congested serving cell.
3. Verify via Simulation: Use tools optimize_antenna_gain(pci, adjust_tilt_angle) on the selected neighbor cell. If the simulation shows a positive gain at the user_location, the solution is validated.
''',

'decrease_a3' : '''
When choosing between "Decrease A3 Offset" solutions, follow this steps:
1. Analyze Temporal Data: Find the timestamp where 5G KPI PCell PCell RF Serving SS-SINR drops below 0dB or Throughput collapses.

2. Identify Source: Record the Serving PCI at that specific timestamp. Cross-reference this PCI with config_data to find the corresponding Cell ID.

3. Identify Target: Look at Measurement PCell Neighbor Cell Top Set. Find the PCI that is consistently stronger than the Serving RSRP during the dip.

4. Validate with Tools (Optional): >    * Use calculate_overlap_ratio(pci_serving, pci_neighbor) to ensure they have overlapping coverage.
    - Use judge_mainlobe_or_not(time, pci_serving); if the UE is outside the serving cell's mainlobe but the neighbor is stronger, a handover is mandatory.

5.Final Selection: Choose the solution that matches the Cell ID of the Source Cell identified in Step 2.

Decision Rule: > * If UE is stuck on Cell A (bad performance) and needs to go to Cell B (strong signal) -> Decrease A3 Offset for Cell A.
''',

'increase_a3' : '''
When choosing between "Increase A3 Offset" solutions to resolve throughput degradation, follow these steps:

1. Identify Culprit Cells (Analyze Three Sources):
    - Signaling: Search signaling_data for NRHandoverAttempt events (ping-pong) occurring rapidly (every 2-5 seconds).
    - User Plane: Check user_plane_data for frequent changes in "Serving PCI" or sharp drops in SINR/Throughput coinciding with specific Neighbor PCIs.
    - Measurement Reports: Search mr_data for Neighbor RSRP values that are stronger than or within 3dB of the Serving Cell (Pilot Pollution).

2. Map and Cross-Reference: Record the PCIs from Step 1 and match them with config_data to identify the corresponding gNodeB ID and Cell ID for the optimization options.

3. Validate with Tools: 
    - Use calculate_overlap_ratio(pci_serving, pci_neighbor): A high ratio (>0.3) confirms overlapping coverage requiring higher hysteresis to prevent "border fluttering."
    - Use judge_mainlobe_or_not(time, pci): If a neighbor cell is triggering measurements while the UE is outside its mainlobe (False), it is an "overshooter" that needs an increased A3 Offset.

4. Final Selection & Filtering: 
    - Include cells involved in confirmed signaling ping-pongs.
    - Include neighbor cells that are consistently stronger than the server or cause interference-driven throughput drops, even if the signaling loop is missing or mismatched.
    - Select the one or two most impactful solutions directly linked to the observed performance degradation.

Decision Rule: 
* Priority: Stabilize the connection by increasing A3 Offset for cells that cause premature handovers OR act as strong interferers.
* Output: Select a maximum of two solutions.
''',

'tilt_down' : '''
When choosing between best "Tilt down" solutions, follow these steps:

1. Analyze Performance Dips & Handover Partners: 
   - Identify the timestamp window where Throughput and SS-SINR collapse.
   - Determine the "Serving Set": List all PCIs that are actively serving or acting as primary handover targets (e.g., if UE is ping-ponging between PCI A and PCI B, both are part of the Serving Set).

2. Identify the Over-shooter (The Aggressor): 
   - Look for a Neighbor PCI with an RSRP within 3-6dB of the serving cell that is NOT part of the "Serving Set."
   - A cell that is consistently strong but fails to take over the session is likely "overshooting" its intended coverage boundary and causing interference.

3. Tool-Based Validation:
   - Call judge_mainlobe_or_not(time, pci_neighbor): If True, the user is outside the cell's mainlobe, confirming it is an over-shooter.
   - Call calculate_overlap_ratio(pci_serving, pci_neighbor): A high ratio indicates significant mutual interference.
   - Call optimize_antenna_gain(time, pci, adjust_tilt_angle): Use this to simulate if the proposed tilt (e.g., 4 or 5 degrees) restores SINR without dropping RSRP below coverage thresholds (-105dBm).

4. Final Selection Logic:
   - SELECT the tilt-down solution for the "Aggressor" PCI (the over-shooter) to shrink its interference footprint.
   - REJECT tilt-down solutions for the current Serving Cell if it would simply weaken the desired signal or shift the handover boundary without fixing the underlying interference.

Decision Rules: 
> * Goal: Stabilize handovers and maximize SINR. The best tilt-down targets "noise," not "service."
> * Mainlobe Rule: Priority is always given to tilting cells where the user is "Outside Mainlobe" (True) yet receiving high RSRP.
''',

'adjust_azimuth' : '''
When choosing between "Adjust the azimuth" solutions, follow these steps:

1. Target the Victim Cell: 
   - Identify the Serving PCI at the exact timestamp of throughput/SINR collapse. 
   - Prioritize adjusting the Serving Cell over neighbors to restore the active link.

2. Verify Beam Deviation:
   - Calculate the bearing from Cell to UE. 
   - The difference between Current Azimuth and Bearing must match the adjustment degrees in the solution (e.g., 47°).

3. Tool-Based Validation:
    - Call calculate_horizontal_angle(time, pci): Determine the exact deviation between the cell's boresight and the UE.
    - Call judge_mainlobe_or_not(time, pci): If True (Outside Mainlobe), this confirms the user is suffering from side-lobe coverage or is at the very edge of the cell's beam.
    - Call optimize_antenna_gain(time, pci, adjust_horizontal_angle, adjust_tilt_angle): Use this to simulate the RSRP gain. Choose the adjustment that provides the highest gain for the serving link.
4. KPI Check:
   - Check traffic_data for "Downlink Weak Coverage Ratio." Prioritize the cell with the highest ratio (>10%), indicating a systemic alignment issue.

Decision Rules: 
> * Primary Goal: Rotate the antenna's main-lobe directly toward the UE coordinates.
> * Hierarchy: Fix Serving Cell > Fix Neighbor Cell.
> * Mathematical Match: Reject any solution where the suggested adjustment does not mathematically close the gap between the current azimuth and the UE's bearing.
''',

'increase_power' : '''
When choosing between "Increase transmission power" solutions to resolve Weak Coverage:
1. Gather Evidence (Tool Phase):
   - Identify Serving PCI at the performance drop timestamp.
   - CALL judge_mainlobe_or_not: If True, the cell is the target.
   - CALL calculate_horizontal_angle: The result must match the adjustment in the candidate option (e.g., 47° gap = 47° adjustment).
   - CALL optimize_antenna_gain: Confirm this adjustment yields the highest RSRP gain.

2. Evaluation (Comparison Phase):
   - Compare tool results against candidates (e.g., C6 vs C22).
   - The "Perfect Match" must: 
     a) Target the failing Serving Cell.
     b) Mathematically close the gap found by calculate_horizontal_angle.
     c) Have a high "Weak Coverage Ratio" in traffic_data.

3. Hard Stop & Selection (Execution Phase):
   - Once tools confirm a candidate matches the physical deviation and provides gain, STOP all tool calls.
   - SELECT the matching candidate immediately.
   - REJECT any option that adjusts a healthy neighbor cell or has a degree value that doesn't match the UE's actual bearing gap.
''',

'decrease_power' : '''
When choosing the best between "Decrease transmission power" solutions, follow these steps:
1.	List all candidate cells
Extract each option’s target cell, such as 3279943_1 or 3267220_2. 
2.	Map each candidate cell to its PCI
Use the configuration data to match (gNodeB ID, Cell ID) to the corresponding PCI. 
3.	Identify bad-throughput samples
In the user-plane data, mark samples where downlink throughput is low, for example: 
o	throughput < 100 Mbps 
4.	Measure how often each candidate is the serving cell during bad samples
For each candidate PCI, count: 
o	how many bad-throughput samples have this PCI as the serving PCI 
5.	Compute serving_bad_ratio for each candidate
serving_bad_ratio = (# bad-throughput samples where candidate PCI is serving) / (total # bad-throughput samples)
6.	Choose the candidate with the lowest serving_bad_ratio
The rationale is: 
o	a cell that is often the serving cell during bad throughput is more likely the victim 
o	a cell that is rarely serving during bad throughput is more likely the interferer 
o	therefore, the best “Decrease transmission power” option is usually the one with the smallest serving_bad_ratio 
7.	Use as a tie-breaker only if needed
If two candidates have very similar serving_bad_ratio, prefer the one that appears more like a persistent strong neighbor in the bad-throughput region.
''' 
}  

SYSTEM_PROMPT = """
You are an expert 5G radio troubleshooting agent for Zindi Track A.

Objective:
- Diagnose throughput degradation from drive-test data.
- Select the best option id(s) from the provided choices (C1..Cn).

Workflow:
1. Find degradation window from throughput logs.
2. Identify serving cell at that window and compare with neighbors.
3. Use signaling/KPI/MR/geometry tools only when needed.
4. Stop when evidence is sufficient and produce the final option id(s).

Notebook-derived hard constraints:
- Use only timestamps that appear in this scenario throughput logs.
- Map candidate options by action category (neighbor, power, threshold, tilt, azimuth, A3, PDCCH).
- For single-answer tasks, return exactly one strongest action id.
- Dependency checks:
  * if choosing decrease_power, verify tilt_down is context-consistent
  * if choosing increase_power, verify adjust_azimuth consistency
- Prefer source-serving-cell ownership for A2/A5 threshold changes.
- Reject options that only look geographically plausible but are unsupported by the degradation window evidence.

Output rules:
- Single-answer tasks: return exactly one ID (e.g. C7).
- Multi-answer tasks: return 2-4 IDs separated by | (e.g. C3|C11).
- Keep answer concise and deterministic.
"""

TOOL_WHITELIST = [
    "get_throughput_logs",
    "get_serving_cell_pci",
    "get_serving_cell_rsrp",
    "get_serving_cell_sinr",
    "get_neighboring_cells_pci",
    "get_neighboring_cell_rsrp",
    "get_signaling_plane_event_log",
    "get_cell_info",
    "get_user_location",
    "get_kpi_data",
    "get_mr_data",
    "judge_mainlobe_or_not",
    "calculate_horizontal_angle",
    "calculate_tilt_angle",
    "calculate_overlap_ratio",
    "optimize_antenna_gain",
]

