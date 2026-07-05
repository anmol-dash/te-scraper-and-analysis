// GAMECA Stage 2 — load/filter TE loci into the family folder (query.py --run-stage data).
//
// Input:  tuple val(meta), path(results), path(input_csv)
// Output: tuple val(meta), path(results)   (now containing 01_data/<family>_raw.csv)

process GAMECA_DATA {
    tag   { meta.id }
    label 'process_low'

    input:
    tuple val(meta), path(results), path(input_csv)

    output:
    tuple val(meta), path("${results}"), emit: results
    path "versions.yml"                , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args      = task.ext.args ?: params.core_args ?: ''
    def assembly  = meta.assembly ?: params.assembly
    def input_arg = (input_csv && input_csv.name != 'NO_FILE') ? "--input ${input_csv}" : ''
    """
    python \$GAMECA_HOME/query.py \\
        --family ${meta.id} \\
        --assembly ${assembly} \\
        --output . \\
        --run-stage data \\
        ${input_arg} ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gameca: \${GAMECA_VERSION:-0.4.42}
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${results}/01_data
    printf 'chr,start,stop,TE_name\\nchr1,1,100,${meta.id}_1\\n' > ${results}/01_data/${meta.id.toLowerCase()}_raw.csv
    echo '"${task.process}": {stub: true}' > versions.yml
    """
}
