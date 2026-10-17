# Original student implementation

`original_course_structure/FEM_LE/` preserves 17 original student Python files and the original final notebook, byte for byte. `manifest.json` records each original relative path, size, SHA-256 hash, and attribution basis. The scalar error routine is copied from the student's broader FEM tree because the final notebook imports it but the focused FEM_LE tree omits it.

The author confirmed that she wrote this Python implementation and reported instructor permission to release her implementation. The Python `TMatrix` and 1D stress-recovery organization follow identifiable MATLAB course references by Professor Kalyana B. Nakshatrala; that lineage is credited in the main documentation. No MATLAB reference file, supplied mesh/table, lecture, other-named notebook, or report containing course diagrams is included here.

This is an archival snapshot, including original notebook outputs, original paths, and known issues. It is not the maintained runnable entry point. Use `examples/` and `src/fem/` for current execution. The three approved corrections and the additional invalid-coarse-SCF restraint case are documented in `docs/methodology.md`. Do not interpret unchanged archived outputs as newly verified results for corrected cases.
