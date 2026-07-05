// GAMECA Stage 6 — visualization dashboard (query.py --run-stage dashboard).
// Runs in parallel with ALIGNMENT/MOTIF/PRIMERS off the same CLUSTERING output;
// GAMECA_MERGE folds all four branches' new files back into one family folder.
//
// Input:  tuple val(meta), path(results)
// Output: tuple val(meta), path(results)   (now containing 07_visualizations/)

process GAMECA_DASHBOARD {
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
        --run-stage dashboard \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gameca: \${GAMECA_VERSION:-0.4.42}
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${results}/07_visualizations
    echo '<html>stub</html>' > ${results}/07_visualizations/index.html
    echo '"${task.process}": {stub: true}' > versions.yml
    """
}
