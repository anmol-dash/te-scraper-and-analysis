// GAMECA Stage 1 — load/pickle-cache the reference genome (query.py --run-stage genome).
// First process in the chain: creates the family output folder.
//
// Input:  tuple val(meta), path(input_csv)        (input_csv may be an empty list)
// Output: tuple val(meta), path(prefix)            the family folder (genome pickle inside)

process GAMECA_GENOME {
    tag   { meta.id }
    label 'process_high'

    input:
    tuple val(meta), path(input_csv)

    output:
    tuple val(meta), path("${prefix}"), emit: results
    path "versions.yml"               , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix     = task.ext.prefix ?: meta.id.toLowerCase()
    def args   = task.ext.args   ?: params.core_args ?: ''
    def genome = params.genome_fasta ? "--genome ${params.genome_fasta}" : ''
    """
    python \$GAMECA_HOME/query.py \\
        --family ${meta.id} \\
        --output . \\
        --run-stage genome \\
        ${genome} ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gameca: \${GAMECA_VERSION:-0.4.42}
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    prefix = task.ext.prefix ?: meta.id.toLowerCase()
    """
    mkdir -p ${prefix}
    echo '"${task.process}": {stub: true}' > versions.yml
    """
}
