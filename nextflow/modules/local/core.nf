// GAMECA core engine (query.py): load a TE family, extract sequences, stats,
// cluster (UMAP/HDBSCAN), dashboard, primers, alignment — i.e. Stages 1-10.
//
// post-alignment analyses ("standout analysis") is deliberately NOT run here: --skip-standout
// tells query.py to stop after the core, and the Nextflow STANDOUT subworkflow
// fans those modules out in parallel instead of query.py's sequential loop.
//
// Composable: `include { GAMECA_CORE } from './modules/local/core.nf'`
// Input  channel: tuple val(meta), path(input_csv_or_empty)
//   meta = [ id: <family>, assembly: <hg38|mm10|...> ]   (nf-core meta convention)
// Output channels:
//   clustered — tuple(meta, <family>_clustered.csv)  the per-copy clustered table
//   results   — tuple(meta, <prefix>/)               the full family output folder

process GAMECA_CORE {
    tag   { meta.id }
    label 'process_high'

    input:
    tuple val(meta), path(input_csv)

    output:
    tuple val(meta), path("${prefix}/01_data/*_clustered.csv"), emit: clustered, optional: true
    tuple val(meta), path("${prefix}")                        , emit: results
    path "versions.yml"                                        , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix        = task.ext.prefix ?: meta.id.toLowerCase()
    def args      = task.ext.args   ?: ''
    def assembly  = meta.assembly ?: params.assembly
    def input_arg = (input_csv && input_csv.name != 'NO_FILE') ? "--input ${input_csv}" : ''
    def genome    = params.genome_fasta ? "--genome ${params.genome_fasta}" : ''
    """
    python \$GAMECA_HOME/query.py \\
        --family ${meta.id} \\
        --assembly ${assembly} \\
        --output ${prefix} \\
        --skip-standout \\
        ${input_arg} \\
        ${genome} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gameca: \${GAMECA_VERSION:-0.4.42}
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    prefix = task.ext.prefix ?: meta.id.toLowerCase()
    """
    mkdir -p ${prefix}/01_data ${prefix}/reports ${prefix}/cluster_alignments
    printf 'Seq,Cluster\\nACGT,0\\nACGA,1\\n' > ${prefix}/01_data/${prefix}_clustered.csv
    echo '"${task.process}": {gameca: stub}' > versions.yml
    """
}
