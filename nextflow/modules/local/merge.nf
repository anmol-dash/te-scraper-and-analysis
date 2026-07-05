// GAMECA_MERGE — folds the four parallel post-clustering branches (DASHBOARD,
// ALIGNMENT, MOTIF, PRIMERS) back into a single family folder before STANDOUT
// (which needs alignment's consensus FASTA + the clustered CSV together).
//
// Each branch started as an identical copy of CLUSTERING's output and only
// added its own new files, so blanket-copying each one on top of the base is
// safe — unrelated files are byte-identical no-op overwrites.
//
// Input:  tuple val(meta), path(base), path(dash), path(align), path(motif), path(primers)
// Output: tuple val(meta), path(base)

process GAMECA_MERGE {
    tag   { meta.id }
    label 'process_low'

    input:
    tuple val(meta), path(base),
          path(dash,    stageAs: 'branch_dashboard'),
          path(align,   stageAs: 'branch_alignment'),
          path(motif,   stageAs: 'branch_motif'),
          path(primers, stageAs: 'branch_primers')

    output:
    tuple val(meta), path("${base}"), emit: results

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    set -euo pipefail
    for d in branch_dashboard branch_alignment branch_motif branch_primers; do
        cp -a "\$d"/. "${base}"/ 2>/dev/null || true
    done
    """

    stub:
    """
    true
    """
}
