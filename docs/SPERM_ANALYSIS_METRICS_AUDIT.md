# Garbha AI — sperm analysis metrics audit

Audit date: 2026-10-08. Read-only inspection of production code, configuration, documentation, saved results and tests; only this report was added. **VERIFIED means the implementation was traced, not that a clinical measurement is validated.**

**Principal findings:** the quoted per-sperm values match an existing result exactly; its CASA values recompute exactly. Morphology has reproducible segmentation, missing-value and aggregation defects. The uploaded-results dashboard is **not present** in this checkout, so its six card calculations cannot all be established. No percentage formula or denominator is invented below.

**Source reference key** — line numbers refer to the inspected working tree; `Class.function:line` identifies the calculation entry point.

- **P**: [sperm_pipeline/sperm_pipeline/pipeline.py](../sperm_pipeline/sperm_pipeline/pipeline.py), `SpermAnalysisPipeline` unless otherwise stated.
- **M**: [morphology.py](../sperm_pipeline/sperm_pipeline/morphology.py), `MorphologyAnalyzer`.
- **C**: [motility.py](../sperm_pipeline/sperm_pipeline/motility.py), `MotilityAnalyzer`.
- **S**: [segmentation.py](../sperm_pipeline/sperm_pipeline/segmentation.py), `SpermSegmentation`.
- **L**: [live_stream.py](../sperm_pipeline/sperm_pipeline/live_stream.py), `LiveStreamProcessor`.
- **A**: [main.py](../main.py), API/configuration functions. **UI**: [static/mobile_camera.html](../static/mobile_camera.html).

**1. Backend-to-frontend trace and populations**

Default uploaded-video path:

1. `P.process_video:201`, specifically 224–226, reads source `CAP_PROP_FPS`. The earlier YOLO analysis pass is diagnostic: its morphology is discarded at line 268. Canonical results come from the subsequent inference pass, not those YOLO tracker IDs.
2. `P._run_maskrcnn_inference:474`, lines 503–516: configured sperm class (currently class **2**), detection score **strictly >0.70**. The additional `>=0.45` gate is redundant. ByteTrack receives only these high-score detections (`P:3191`); only matched tracks with processed mask area **>=6 pixels** reach the identity linker (`P:3201–3229`). Unmatched, ambiguous and unresolved observations can therefore be excluded even at high confidence.
3. `P._link_inference_application_ids:2116`: canonical Application IDs, one assignment per identity per frame; new IDs only after recovery/deferral gates. Scores **0.30–0.70 inclusive** can support an existing ID, but cannot create one or supply uploaded morphology/CASA. `morphology_valid` means confidence eligibility, **not successful anatomical assessment**.
4. `P._generate_inference_video:3128` (body at 3312–3336): every fifth source frame by default, accepted eligible masks supply morphology under the same Application ID. `P._trajectories_from_inference_rows:3398` excludes `morphology_valid=false` and builds `(frame,(x1+x2)/2,(y1+y2)/2)`. Coordinates are **ByteTrack-associated bbox centers**, not head centroids or the separate CSV mask-centroid fields.
5. `P._compute_final_analysis:3483` emits one result per canonical trajectory key, including short tracks and tracks lacking morphology. `sperm_id == application_id`. `P._get_morphology_summary:3563` uses all available morphology samples; CASA uses the last one second of that track’s observations, not a whole-video average.
6. `P._save_results:3618` writes `meta/summary.json`, compact CSV and source-FPS `trajectories.json`. `A.process_video_async:189` stores the list; `A.get_results:316`, lines 360–375, returns it unchanged in `results`, with artifact links and `video_properties`. It adds **no percentage aggregation**. `morphology_score` is exported, but a separate track-level `morphology_status` is not; top-level `status` combines morphology and motility.
7. The referenced `templates/index.html` (`P:3536`) is absent. The only checked-in frontend is the live mobile page. `UI.updateStats:474–479` displays received counts without percentages: `total_tracked`, `active_count`, `motility.progressive`, `fps_actual`. The uploaded dashboard’s formatting, denominator choices and unit bindings are **NOT VERIFIED**.

Let **T** be the keys returned by `P._uploaded_trajectories:3445`. Uploaded result-row count is `len(T)`, not detector observations, trajectory-point count, maximum sperm ID, or a validated physical-cell count. Fragmentation can increase `len(T)`; missed/unresolved sperm can reduce it. IDs are video-local, monotonic and not recycled (`P.SpermIdentityRegistry.allocate_id:31`). There is no morphology-assessable denominator or biological deduplication after tracking.

Alternate paths differ: CSRT exports canonical IDs but uses `require_morphology_valid=False` (`P:903–905`), including validated tracking updates between detector corrections; detector segments use configurable `score > threshold` (default 0.70; `S:236`). Live uses YOLO confidence **0.25** (`detection.SpermDetector.__init__:22`, `_detect_single:87`) and session-local `SpermTracker`, not uploaded canonical IDs.

**2. Dashboard metric summary**

For absent percentage implementations, numerator, denominator, confidence filtering and missing-data policy are all unknown at the dashboard layer. The populations above describe available backend inputs, not an inferred frontend formula.

| Metric | Actual formula / logic | Threshold | Unit | Source file/function | Correctness status |
|---|---|---|---|---|---|
| Total Motility (%) | No uploaded percentage calculation found; numerator/denominator and empty-population behavior unknown | Dashboard unknown | % label unverified | A `get_results`:316; UI `updateStats`:474 | NOT VERIFIED |
| Progressive Motility (%) | No uploaded percentage calculation found. Mobile “Progressive” is an integer, not a percentage | Dashboard unknown; per-track rules below | % label unverified | A:374; L `get_motility_summary`:341; UI:477 | NOT VERIFIED |
| Normal Morphology (%) | No percentage calculation found. Internal track normality is `median(frame scores) > 0.5`, not “all criteria pass” | >0.5; no assessability filter | % label unverified | P `_get_morphology_summary`:3563, especially 3599 | NOT VERIFIED |
| Total Cells Detected | Card binding absent. Backend offers result rows, registry-created IDs, assigned observation rows and live counts; which one the card uses is unknown | Dashboard unknown | Count | P:3398, 3513, 1872; A:374 | NOT VERIFIED |
| Good Quality | Per-track `status='good'` iff `morphology_summary.status == 'normal' AND motility.label == 'progressive'`; otherwise `defective`. No aggregate card count implemented here | Median morphology score >0.5 plus progressive rule | Boolean/status; card count unknown | P `_compute_final_analysis`:3521–3524 | BUG |
| Unique tracked sperm / total sperm count | Uploaded population: one row per key of T. Default tracking diagnostic `application_ids_created = count(ID_CREATED events)`; live `total_tracked = len(all trajectories)`. No physical count estimator or separate uploaded scalar total | Default uploaded >0.70; live YOLO 0.25; no track-length exclusion from uploaded rows | IDs/tracks, not validated cells | P:3513, `_build_tracking_quality_summary`:1924–1927; L:355 | VERIFIED |
| Morphology score | Frame score `clamp(1 − 0.2 × number_of_failed_checks,0,1)`; track score is median of frame scores | Five checks below | 0–1 | M `_classify_morphology`:210; P:3591–3603 | VERIFIED |
| Motility score | Progressive →1; non-progressive →0.5; all other labels →0 | Label lookup, not a physical measurement | 0–1 | P:3531 | VERIFIED |
| Mobile Tracked / Active / Progressive | `len(all trajectories)` / `len(active trajectories)` / count of progressive active tracks having >=5 points | Live confidence 0.25; final live summary instead assesses >=3 points | Counts | L:193–215, 341–359, 412–433; UI:474–479 | VERIFIED |
| Mobile Server FPS | `round(1/max(processing_elapsed_seconds,0.001),1)`; not measured camera capture rate or CASA time base | None | frames/s | L:433; UI:479 | VERIFIED |

“Good” is an application-specific conjunction, **not fertility potential or clinical sperm quality**. Its BUG status reflects the contradictory normality aggregation demonstrated below. None of these custom classification cutoffs is established as WHO/Kruger criteria by the inspected implementation.

**3. Morphology formulas, thresholds and missing parts**

All five displayed morphology measurements are per-sample measurements followed by **independent per-track medians** (`P:3586–3588`). No microscope scale is passed to `MorphologyAnalyzer` (`P:152–155`); lengths remain pixels even when CASA is calibrated.

| Metric | Actual formula / logic | Threshold | Unit | Source file/function | Correctness status |
|---|---|---|---|---|---|
| Head Length | Largest 8-connected component of supplied head mask; `region.major_axis_length` from second-moment ellipse. Bbox longer side only if two inertia eigenvalues unavailable | Normal interval [5,60] | px | M `_analyze_head`:85; 104–131 | POTENTIAL ISSUE |
| Head Width | Same component, `region.minor_axis_length`; bbox shorter side only in fallback | Normal interval [3,40] | px | M `_analyze_head`:124–131 | POTENTIAL ISSUE |
| Head Circularity | `4πA/P²`, A=`region.area`, P=`region.perimeter`; substitute P=1 if perimeter<=0. Not bbox area/perimeter or a separately extracted contour | >=0.70 | Dimensionless | M `_analyze_head`:119–121 | POTENTIAL ISSUE |
| Head area | Pixel area of selected component; calculated internally but omitted from final five-feature median dictionary | No classification cutoff | px² | M:119, 69; P:3586 | VERIFIED |
| Tail Length | Number of foreground pixels in `skeletonize(tail_mask)`; includes all components/branches, without diagonal-length weighting or endpoint tracing | >=15 | Skeleton-pixel count, labeled px | M `_analyze_tail`:143–160 | POTENTIAL ISSUE |
| Neck Angle | Absolute wrapped angle between head-centroid→furthest-tail-pixel vector and first SVD principal axis of all tail pixels. **Neck mask and head orientation are not used** | <=45 | degrees, 0–180 | M `analyze_morphology`:62; `_compute_neck_angle`:162–208 | BUG |
| Head length/width ratio | Not calculated, exported or tested against a threshold | None | — | M:65–73; P:3586 | NOT IMPLEMENTED |
| Acrosome coverage | No segmentation, area fraction or classification | None | — | S:327–329; M:65–73 | NOT IMPLEMENTED |
| Vacuoles | No detection, count or area measurement | None | — | S:327–329; M:65–73 | NOT IMPLEMENTED |
| Residual cytoplasm | No measurement or classification | None | — | S:327–329; M:65–73 | NOT IMPLEMENTED |
| Normal morphology, per sample | `normal` iff **zero** failed checks; otherwise `defective` | All five checks must pass | Status | M `_classify_morphology`:254 | VERIFIED |
| Normal morphology, per track | `normal` iff median sample score >0.5; can accept one/two defects in every sample | >0.5, not all-checks-pass | Status | P `_get_morphology_summary`:3599 | BUG |

Length/width normally follow the rotated component’s **principal axes**, not an axis-aligned bounding box. That does not ensure the component is actually a head.

**Mask provenance:** default uploaded morphology uses a temporally smoothed full-sperm mask: current weight 0.65 when >50 high detections, otherwise 0.75; previous mask translated by centroid motion; binary threshold >=0.50; 3×3 elliptical closing once; bbox crop with 3-pixel padding (`P:592–628, 3194–3229, 3429–3443`). Closing can join nearby structures.

`S.segment_subparts:300` uses HNK softmax/argmax classes 1=head, 2=neck, 3=tail, **without a subpart-confidence gate or intersection with the supplied sperm mask**. On unavailable/non-callable/failed HNK inference, it uses the heuristic (`S:169–187, 311–339`). The heuristic labels the **most circular connected component of the full sperm mask as the entire head** and defines tail as full-mask skeleton outside that component (`S:354–384`). For one connected sperm, head equals the whole component and tail is empty. Historical `uvicorn.log:6` explicitly records the HNK state-dict/non-callable fallback; per-result fallback provenance is not exported.

**Issue-label generation:** each failure subtracts 0.2 and appends the literal label. Constructor-configurable parameters exist (`M:20–40`), but the production pipeline supplies defaults; these five cutoffs have no config.ini/API wiring and are not micrometer-calibrated or verified WHO/Kruger thresholds.

| Issue label | Exact trigger | Missing-part behavior | Source |
|---|---|---|---|
| low head circularity | circularity <0.70 | Empty head →0 →label | M:95–102, 223–227 |
| abnormal head length | length <5 or >60 | Empty head →0 →label | M:230–233 |
| abnormal head width | width <3 or >40 | Empty head →0 →label | M:236–239 |
| excess neck angle | angle >45 | Missing tail, <2 tail pixels or failed SVD →0 →**no label** | M:174–200, 242–245 |
| short tail length | tail length <15 | Empty tail →0 →label | M:153–158, 248–251 |

Final reasons are the **deduplicated union over every sampled frame**, not checks rerun on the displayed median features (`P:3592–3597`). Missing all morphology gives score 0, status defective, reason `no morphology data`, and flattened numeric zeros (`P:3573–3579, 3537–3541`). Missing anatomy is not represented as “unassessable.”

**4. Motility/CASA formulas and classification**

Let selected observations have frame indices fᵢ and bbox centers pᵢ. Keep observations with `fᵢ >= last_frame − FPS×1.0`. Set qᵢ=pᵢ without calibration, otherwise qᵢ=pᵢ×s, where s is measured µm/pixel. Let `Δt=(f_last−f_first)/FPS`; for zero frame span use `1/FPS`. Gaps contribute elapsed time; no missing points are interpolated (`C:86–102, 132–162, 218–235`).

| Metric | Actual formula / logic | Threshold | Unit | Source file/function | Correctness status |
|---|---|---|---|---|---|
| VCL | `Σ ||qᵢ₊₁−qᵢ|| / Δt` | No formula cutoff | px/s or µm/s | C `_calculate_motility_metrics`:218–226 | VERIFIED |
| VSL | `||q_last−q_first|| / Δt` | Not used directly for classification | px/s or µm/s | C:221–227 | VERIFIED |
| VAP | Path length of smoothed q divided by Δt. Interior point averages actual samples within ±0.1 seconds of its frame; endpoints fixed; <3 selected points unchanged | Smoothing 0.2 s total width | px/s or µm/s | C `_smooth_average_path`:141; metrics:229–232 | VERIFIED |
| LIN | `VSL/VCL` when VCL>1e−6, else 0; `LIN_percent=100×LIN` | Progressive LIN>=0.70 | Ratio; separate % field | C:184, 235 | VERIFIED |
| Path deviation | Mean Euclidean distance from qᵢ to corresponding smoothed point; not used in classification and not ALH | None | px or µm; no dedicated unit field | C:233; `_result`:185 | VERIFIED |
| STR, WOB, ALH, BCF | No calculation or result fields | None | — | C `_calculate_motility_metrics`:202; `_result`:179–196 | NOT IMPLEMENTED |
| Immotile | First: original trajectory <3 points or selected window <2 points →immotile with numeric zeros and reason. Otherwise `net_displacement<2 AND VCL<5` | Strict <2 and <5 | Label | C `analyze_motility`:86–96; `_classify_motility`:265 | BUG |
| Progressive | After early returns/immotile test: `VCL>=5 AND LIN>=0.7` | Inclusive 5 and 0.7 | Label | C `_classify_motility`:270 | POTENTIAL ISSUE |
| Non-progressive | Every remaining assessed case, including VCL<5 when displacement>=2 | Residual branch | Label | C:274 | POTENTIAL ISSUE |
| Total motility | No sample-level total-motility formula or percentage implemented | No denominator defined | — | A `get_results`:360–375 | NOT IMPLEMENTED |

VCL/VSL arithmetic is verified on the implemented bbox trajectory, **not on validated sperm-head trajectories**. VAP is explicitly application-defined, not a claimed universal CASA smoother. Minimum three points is checked **before** windowing: an initially longer track can legitimately leave two points and then yield VAP=VCL.

**Units/calibration:** `config.ini:29` is blank; `A._cfg:55` permits environment override `SPERM_PIPELINE_MICROMETERS_PER_PIXEL`; `A:98–99, 165` wires it to the pipeline. Missing scale gives **px/s plus an uncalibrated warning**, not an exception. Positive finite scale gives µm/s; invalid supplied scale or FPS raises (`C:63–72, 117–130, 173–196`; `P:3495–3510`). Morphology never receives this scale. Runtime configuration for every deployment is not established by the checked-in blank value.

Classification cutoffs **2 and 5 remain historical pixel numbers even after coordinates are converted to micrometers**. No threshold conversion or validated physical-unit configuration exists (`C:53–55, 98–105, 263–270`). Thus calibrated output can have inconsistent classification semantics. These are application thresholds, not verified WHO/CASA clinical classification criteria.

Uploaded CASA uses actual **source-video metadata FPS**, not a hardcoded 30 FPS; variable-frame-rate capture timestamps are not used. Live uses configured `target_fps` (default 30), not actual capture timing (`L:198–202, 344–348, 419`). The unused helper `P._identify_good_quality_sperm:3451` hardcodes 30 FPS, but is not called by `process_video` and does not explain uploaded results.

**5. The quoted dashboard example — matched evidence**

All quoted values match sperm ID **1** in [saved summary.json, row beginning line 104](../outputs/a862cc35-1572-4ada-9ece-dd63e62bf946/meta/summary.json#L104), also present in `outputs/identity_registry_validation/final_primary/meta/summary.json`. The screenshot’s specific job URL was not supplied; identical saved values do not establish which copy it displayed.

| Question | Finding |
|---|---|
| Why length 172.3 versus width 36.6? | Stored values are 172.277915 and 36.558009 px, independently aggregated medians. Their ratio is about 4.71, but ratio is not an implemented feature. The whole-connected-component-as-head fallback provides a concrete mechanism for an elongated false “head.” |
| Does head include neck/tail/other structures? | The fallback necessarily includes everything connected in the selected component; learned HNK also lacks a crop-mask intersection. The exact historical per-frame head masks/provenance were not saved in this summary, so contamination of this specific row is strongly plausible, not directly proven. |
| Is circularity 0.138 calculated correctly? | Stored 0.138435785 is consistent with the implemented `4πA/P²` measure, aggregated by median. The formula is correct for its supplied component; wrong anatomy or a ragged mask makes it an invalid head measurement. Raw per-frame A/P are absent, so this exact morphology value cannot be independently recomputed from summary alone. |
| Does tail 0 mean anatomical zero? | No. Empty inferred tail returns 0; median 0 can arise from zero measurements in at least half the samples. Connected-mask fallback creates precisely this condition. It does not demonstrate an absent biological tail. |
| Does neck 0 mean normal alignment? | No. Missing/insufficient tail and SVD failure all return 0. Neck mask is ignored. The displayed median cannot distinguish unavailable from aligned. |
| Actual source FPS for velocities? | Saved trajectories use **16.129032258 FPS**. Of 88 points, frames **71–87** enter the final window: Δt=16/FPS=**0.992 s**. Recomputing from that saved trajectory gives exactly the stored VCL/VSL/VAP/LIN, maximum absolute error **0.0**. |
| Is LIN VSL/VCL? | Yes: 87.52353108576101 / 94.31379196370445 = **0.9280035216847543**; LIN_percent=92.800352%. |
| Why progressive? | VCL=94.313792>=5 and LIN=0.928004>=0.7. VAP=90.637713 does not enter classification. Units are explicitly **px/s**, calibration is null. |
| Why contradictory issue labels? | Width median 36.6 is within [3,40] and angle median 0 is <=45, but reasons retain failures from other samples. Labels can also arise from fallback/zero measurements. They are not generated by reevaluating the displayed medians. |
| Is this sperm Good Quality? | **No:** saved morphology_score≈0.4, motility_score=1, overall status=`defective`. Progressive alone is insufficient. |

**6. Denominator and aggregation audit**

- Uploaded rows retain tracks with only one/two observations: they become “immotile,” not excluded/unassessable. Consequently any external percentage over all rows can count insufficient evidence as immobility. No such frontend percentage is verifiable here.
- Missing morphology does not remove a row: it becomes defective with zeros. Conversely normality uses median score, allowing persistent failures; no `assessable` flag exists. Morphology samples every fifth frame and CASA uses high-confidence positions, so evidence coverage differs even when the row IDs agree.
- Unassigned/unresolved observations are excluded from canonical trajectories/results, while accepted low-confidence support appears in inference CSV but not default CASA. Thus CSV row count, trajectory-point count and result-row count are different quantities.
- Live final summary denominator candidate `total_tracked` includes **all** tracks, but `per_sperm`/class counts include only tracks with >=3 points. Live overlay counts only **active** tracks with >=5 points. Dividing either population by all tracked identities would not be an assessed-sperm percentage; the mobile frontend does not perform this division.
- Identity fragmentation can duplicate physical sperm across IDs; no post-tracking biological merging/count correction is performed. A maximum ID must never be reported as a physical count, even when its number happens to equal allocated-ID count.

**7. Confirmed bugs and limitations — recommended corrections only**

Severity indicates reporting impact, not a clinical risk validation. No corrections were applied.

| Issue | Current behavior | Expected behavior | Severity | Recommended correction |
|---|---|---|---|---|
| BUG: whole sperm treated as head | Single connected fallback mask becomes head; its tail is empty (`S:379–384`) | Anatomical head/tail segmentation or explicitly unavailable morphology | High | Load/validate HNK architecture and class mapping; mark fallback measurements unassessable rather than anatomical. |
| BUG: inconsistent normality / Good Quality | A sample with missing tail scores 0.8 and is defective, but track median 0.8 becomes normal and can be good (`M:254`, `P:3599,3524`) | Consistent, explicit normality criteria and missing-data handling | High | Separate score from normality; use a documented valid-sample aggregation and assessability gate. |
| BUG: missing values look measured | Empty head/tail or missing morphology →zeros; unavailable neck can pass its check | Distinguish zero from unavailable | High | Return null plus part-validity/reason flags; exclude unavailable measurements from anatomical thresholds and explicitly report denominators. |
| BUG: neck angle lacks anatomical/sign invariance | Ignores neck/head axis; SVD axis sign can make an aligned tail 0° or 180° (`M:162–208`) | Orientation-invariant, anatomically defined angle | High | Define head/neck/tail reference axes and resolve axis sign; return unavailable on insufficient anatomy. |
| BUG: calibrated classification uses pixel cutoffs | Coordinates scale to µm but cutoffs remain 2/5 (`C:98–105,263–270`) | Thresholds and measured quantities in consistent declared units | High | Separate pixel exploratory classification from validated physical-unit thresholds; do not simply relabel numbers. |
| BUG: insufficient trajectories called immotile | <3 total points or <2 window points yields zero velocities/immotile (`C:87–96`) | Unassessable with explicit reason | High | Introduce insufficient-data status and defined inclusion rules for downstream percentages. |
| BUG: CSRT fallback receives probability mask | `P._analyze_morphology_for_frame:367–370` passes floats; fallback casts to uint8 (`S:355`), turning values <1 into 0 | Consistent binary-mask input contract | High in CSRT mode | Explicit probability threshold before fallback; retain original probabilities separately. Default canonical path already supplies a binary mask. |
| POTENTIAL ISSUE: live timing/calibration | CASA uses target FPS; live analyzer is constructed without spatial scale (`L:50,419`) | Acquisition timestamps and applicable measured scale | High for physical velocity reporting | Pass calibrated setup and real frame times; retain px/s warning until calibrated. |
| POTENTIAL ISSUE: bbox-based CASA / tail length | Bbox shape changes contribute motion; tail length counts skeleton pixels including branches | Validated head position and anatomical path-length measurement | Medium–high | Validate localization; distinguish skeleton-pixel count from calibrated centerline length. |
| POTENTIAL ISSUE: reasons versus medians | Any-frame issue survives beside passing median (`P:3586–3599`) | Explicit temporal explanation | Medium | Report issue frequency/valid sample count, or label reasons “observed in one or more frames.” |
| NOT VERIFIED: dashboard percentages/count labels | Uploaded frontend absent; no backend percentage contract | Named numerator, denominator, exclusions and units | High reporting gap | Locate and audit the uploaded-results dashboard, then bind cards to an explicit backend summary contract. Do not substitute guessed formulas. |
| BUG: stale documentation | Existing CASA document says missing calibration fails/returns unknown; current code returns px/s | Documentation agrees with implementation | Medium | Update that document separately; also correct its claim that windowing cannot leave two points. |

**8. Validation evidence and scope**

Read `README.md`, `docs/MOTILITY_CASA_CALCULATIONS.md`, detection/segmentation/morphology/motility/tracking/pipeline/live/API/frontend/configuration and relevant tests. Current code takes precedence over older contradictory prose. No outside WHO reference was used to invent or endorse a cutoff.

Existing suite: **94 passed in 7.35 s**, run with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sperm_pipeline /home/user_garbha/sperm/bin/python3 -m pytest -q -p no:cacheprovider --basetemp=/tmp/sperm_metrics_audit_pytest`. No production changes or new test files. Relevant assertions: [test_casa_metrics.py:47](../tests/test_casa_metrics.py#L47) (velocity arithmetic), :67 (scale), :91 (frame gaps), :101/:114 (px/s without calibration), :135 (invalid FPS); [test_uploaded_identity_contract.py:6](../tests/test_uploaded_identity_contract.py#L6) (low-support exclusion), :27 (canonical IDs); [test_inference_identity_linker.py:429](../tests/test_inference_identity_linker.py#L429) (diagnostic counts). Passing these tests does not validate clinical morphology.

Read-only in-memory probes additionally reproduced:

- Connected synthetic head+tail mask: **707/707 pixels assigned to head**, tail 0, head length 179.752 px and circularity 0.09014; neck 0.
- Same mask with probability 0.9: heuristic uint8 conversion returns **all empty parts**.
- Straight horizontal tail, head at opposite aligned ends: **0° versus 180°**.
- Otherwise passing morphology with tail 0: frame **defective**, score **0.8**, track **normal**, final **good** when progressive.
- Identical three-point trajectory: uncalibrated classification **progressive**, supplied scale 0.1 classification **immotile**, demonstrating unit-sensitive unchanged numeric cutoffs.
- Quoted saved sperm’s four CASA values recomputed exactly using saved source FPS and trajectory. Historical raw subpart masks and the uploaded dashboard implementation remain unavailable for verification.

**9. Plain-language explanation**

- **Total Motility:** This service does not calculate the displayed percentage; its dashboard formula and assessed population still need verification.
- **Progressive Motility:** A track is called progressive when speed is at least 5 in the current coordinate units and straightness ratio is at least 0.7. The dashboard percentage is not present here.
- **Normal Morphology:** Internal track normality means median morphology score above 0.5, which currently can tolerate persistent defects. The displayed percentage cannot be verified.
- **Total Cells Detected:** The card’s source is unknown. Available uploaded records represent tracked identities, not a validated number of physical sperm.
- **Unique tracked sperm:** One canonical identity record per uploaded trajectory; one sperm can still produce multiple records if tracking fragments.
- **Good Quality:** The application requires its track-level “normal” morphology and progressive movement together; this is not a clinical fertility judgment.
- **Head Length / Width:** Principal-axis dimensions of the supplied head component in pixels. Incorrect subpart segmentation can measure much more than the head.
- **Head Circularity:** A shape ratio from area and perimeter; small values describe an elongated/irregular supplied mask, not necessarily an abnormal biological head.
- **Tail Length:** Count of pixels on the inferred tail skeleton. Zero often means no tail mask was obtained.
- **Neck Angle:** A tail-direction proxy with a confirmed orientation defect; zero can mean missing data, not normal alignment.
- **VCL / VSL / VAP:** Speed along the observed bbox path / directly between endpoints / along an averaged path, over the track’s last second of observations.
- **LIN:** VSL divided by VCL; 0.928 means about 92.8% linearity. It is not a speed.
- **Non-progressive / Immotile:** Remaining movement classes under the custom rules; insufficient observations are currently mislabeled immotile.
- **Morphology / Motility scores:** Application scores, not probabilities of fertility or calibrated clinical quality.
- **Head ratio, acrosome, vacuoles, residual cytoplasm, STR, WOB, ALH and BCF:** Not implemented in this repository.
- **Mobile Tracked / Active / Progressive / Server FPS:** All session tracks / active tracks / progressive active tracks with enough observations / server processing rate; these are not uploaded sample percentages.
