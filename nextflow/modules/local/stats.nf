// GAMECA Stage 4 — basic statistics (query.py --run-stage stats).
//
// Input:  tuple val(meta), path(results)
// Output: tuple val(meta), path(results)   (now containing 02_statistics/ + expr_cols.json)

process GAMECA_STATS {
    tag   { meta.id }
    label 'process_low'

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
        --run-stage stats \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gameca: \${GAMECA_VERSION:-0.4.42}
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${results}/02_statistics
    echo '[]' > ${results}/01_data/expr_cols.json
    echo '"${task.process}": {stub: true}' > versions.yml
    """
}
