// GAMECA Stage 7 — MSA + CIAlign consensus building (query.py --run-stage alignment).
// Runs in parallel with DASHBOARD/MOTIF/PRIMERS off the same CLUSTERING output;
// GAMECA_MERGE folds all four branches' new files back into one family folder.
//
// Input:  tuple val(meta), path(results)
// Output: tuple val(meta), path(results)   (now containing cluster_alignments/, cleaned_consensus/, cialign_plots/)

process GAMECA_ALIGNMENT {
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
    def args = task.ext.args ?: params.core_args ?: ''
    """
    python \$GAMECA_HOME/query.py \\
        --family ${meta.id} \\
        --output . \\
        --run-stage alignment \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gameca: \${GAMECA_VERSION:-0.4.42}
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${results}/cluster_alignments
    echo '>consensus' > ${results}/cluster_alignments/all_cluster_consensuses.fa
    echo '"${task.process}": {stub: true}' > versions.yml
    """
}
