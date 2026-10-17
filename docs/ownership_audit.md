# Ownership and redistribution audit

The original FEM folder was inventoried before changes: **351 files** across `HW`, `HW_Soln`, `Slides`, `Videos`, and `pictures`. The file-level [CSV inventory](ownership_audit.csv) records the classification, inclusion decision, and evidence for every file. ZIP member lists were also inspected to identify duplicates, cached binaries, and mixed materials. Source files, notebook sources and saved outputs, MATLAB headers, relevant reports, and provided-input metadata were reviewed. Lecture videos were classified from their location and names; their full audiovisual content was not reviewed or copied.

The author explicitly confirmed that she wrote the Python implementation under `HW_Soln/FEM_LE` herself and reported verbal instructor permission to release her implementation. Most Python headers and the authored project report support that attribution. This confirmation is recorded as the basis for including the student Python code; no independent ownership determination is implied.

## Classification and release decisions

| Category | Evidence and treatment |
| --- | --- |
| Student-authored code | The Python framework, elasticity kernel, mesh/experiment notebook, and error routines have author confirmation and supporting headers/report descriptions. Canonical relevant files are archived unchanged. |
| Student-authored code with reference lineage | `TMatrix.py` and `Stress_Recovery.py` have identifiable instructor MATLAB counterparts. They are credited as student Python implementations following course reference formulations/organization, not as independent inventions. The MATLAB files themselves are excluded. Other interfaces follow course specifications, also credited. |
| Instructor-provided reference/starter code | `HW/matlab/`, `HW/matlab.zip`, and the Gauss-point text files. MATLAB headers explicitly name Professor Kalyana B. Nakshatrala where present. Excluded from the release. |
| Supplied inputs | `HW/Square_plate_with_hole_Q4.geo`, `.msh`, and `.mat`. Redistribution permission is not established. Excluded; maintained examples construct their own inputs. |
| Course / third-party materials | Lecture slides, videos, textbook reading, course handouts, and homework solution PDFs. Excluded. |
| Other-named student work | `Taiwo_Adebiyi_CIVE_7336_MP_2_Final.ipynb` is outside the confirmed FEM_LE implementation scope. Excluded; no content is copied into the release. |
| Reports with mixed material | The author's reports include course diagrams or other course-derived material. They remain in the local source folder and are excluded from the public archive. Numerical conclusions were audited against the runnable code and saved notebook outputs. |
| Earlier/duplicate work and generated data | The broader `FEM` tree includes diffusion work, duplicate elasticity code, checkpoints, data, and generated results. The canonical elasticity implementation is selected instead. Bytecode and complete mixed-content ZIPs are excluded. |
| Unresolved images | The `pictures` folder is excluded because individual ownership and redistribution rights are not established. |

No file is included merely because it is stored in a student-named folder or ZIP. The release includes **17 original student Python files and one original final notebook**, with hashes in [archive/manifest.json](../archive/manifest.json). One scalar error routine required by the final notebook is taken from the matching broader student `FEM/Error_Analysis` tree because it is absent from the focused FEM_LE copy. The two final-notebook versions have identical cell sources but different saved outputs; the FEM_LE version is the provenance source.

The original report includes an AI-use declaration. The student remains the author of the implementation as confirmed by her; the later public-repository refactoring and verification were AI-assisted. No instructor source is attributed to her, and no license has been added.

This audit documents provenance and a conservative file-selection decision; it does not expand the scope of the instructor's reported permission. The original course source files remain unchanged; the release is placed in a separate subdirectory.
