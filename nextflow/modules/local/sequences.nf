// GAMECA Stage 3 — fetch/extract sequences (query.py --run-stage sequences).
// Reloads the genome pickle written by GAMECA_GENOME (cheap vs. re-parsing the FASTA).
//
// Input:  tuple val(meta), path(results)
// Output: tuple val(meta), path(results)   (now containing 01_data/<family>_with_sequences.csv)

process GAMECA_SEQUENCES {
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
    def args     = task.ext.args ?: params.core_args ?: ''
    def assembly = meta.assembly ?: params.assembly
    def genome   = params.genome_fasta ? "--genome ${params.genome_fasta}" : ''
    """
    python \$GAMECA_HOME/query.py \\
        --family ${meta.id} \\
        --assembly ${assembly} \\
        --output . \\
        --run-stage sequences \\
        ${genome} ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gameca: \${GAMECA_VERSION:-0.4.42}
    END_VERSIONS
    """

    stub:
    """
    cp ${results}/01_data/${meta.id.toLowerCase()}_raw.csv \\
       ${results}/01_data/${meta.id.toLowerCase()}_with_sequences.csv
    echo '"${task.process}": {stub: true}' > versions.yml
    """
}
