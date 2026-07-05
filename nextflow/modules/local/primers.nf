// GAMECA Stage 10 — primer design (query.py --run-stage primers).
// Reloads the genome pickle written by GAMECA_GENOME for genome-wide search.
// Runs in parallel with DASHBOARD/ALIGNMENT/MOTIF off the same CLUSTERING output;
// GAMECA_MERGE folds all four branches' new files back into one family folder.
//
// Input:  tuple val(meta), path(results)
// Output: tuple val(meta), path(results)   (now containing 06_primers/)

process GAMECA_PRIMERS {
    tag   { meta.id }
    label 'process_high'

    input:
    tuple val(meta), path(results)

    output:
    tuple val(meta), path("${results}"), emit: results
    path "versions.yml"                , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args ?: params.core_args ?: ''
    def genome = params.genome_fasta ? "--genome ${params.genome_fasta}" : ''
    """
    python \$GAMECA_HOME/query.py \\
        --family ${meta.id} \\
        --output . \\
        --run-stage primers \\
        ${genome} ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gameca: \${GAMECA_VERSION:-0.4.42}
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${results}/06_primers
    echo '"${task.process}": {stub: true}' > versions.yml
    """
}
