// GAMECA Stage 5 — UMAP/HDBSCAN clustering (query.py --run-stage clustering).
//
// Input:  tuple val(meta), path(results)     (needs 01_data/<family>_with_sequences.csv)
// Output: tuple val(meta), path(results)     (now containing 01_data/<family>_clustered.csv)

process GAMECA_CLUSTERING {
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
        --run-stage clustering \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gameca: \${GAMECA_VERSION:-0.4.42}
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${results}/03_clustering
    printf 'Seq,Cluster\\nACGT,0\\nACGA,1\\n' > ${results}/01_data/${meta.id.toLowerCase()}_clustered.csv
    echo '"${task.process}": {stub: true}' > versions.yml
    """
}
