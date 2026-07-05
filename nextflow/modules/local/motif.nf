// GAMECA Stage 9 — JASPAR motif / TFBS / GO analysis (query.py --run-stage motif).
// Runs in parallel with DASHBOARD/ALIGNMENT/PRIMERS off the same CLUSTERING output;
// GAMECA_MERGE folds all four branches' new files back into one family folder.
//
// Input:  tuple val(meta), path(results)
// Output: tuple val(meta), path(results)   (now containing motif_analysis/, enrichment_results/, go_annotations/, 05_tfbs/)

process GAMECA_MOTIF {
    tag   { meta.id }
    label 'process_medium'

    input:
    tuple val(meta), path(results)

    output:
    tuple val(meta), path("${results}"), emit: results
    path "versions.yml"                , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args     = task.ext.args ?: params.core_args ?: ''
    def assembly = meta.assembly ?: params.assembly
    """
    python \$GAMECA_HOME/query.py \\
        --family ${meta.id} \\
        --assembly ${assembly} \\
        --output . \\
        --run-stage motif \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gameca: \${GAMECA_VERSION:-0.4.42}
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${results}/motif_analysis
    echo '"${task.process}": {stub: true}' > versions.yml
    """
}
