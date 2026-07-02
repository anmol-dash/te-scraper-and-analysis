// GAMECA post-alignment analyses "standout analysis" — one module invocation per task.
//
// query.py runs the ~15 run_*.py modules sequentially in a subprocess loop
// (see query.py `_run_standout_analysis` / `_MODULES`). This process runs a
// SINGLE module; the STANDOUT subworkflow scatters the registry across many of
// these so they execute in parallel, each with its own LSF resource request.
//
// Each module reads the clustered CSV + consensus FASTAs from the staged family
// folder and writes its artifacts into an isolated `<name>_reports/` dir, which
// GAMECA_REPORT later merges back into the family's reports/ folder.
//
// Input: tuple val(meta), path(results), val(mod)
//   mod = [ name: 'phylo', script: 'run_phylo_analysis.py', args: '--assembly hg38 ...' ]

process GAMECA_STANDOUT {
    tag   { "${meta.id}:${mod.name}" }
    label 'process_medium'

    input:
    tuple val(meta), path(results), val(mod)

    output:
    tuple val(meta), path("${mod.name}_reports"), emit: reports
    path "versions.yml"                         , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def clustered = "${results}/01_data/${meta.id.toLowerCase()}_clustered.csv"
    def cons      = "${results}/cluster_alignments/all_cluster_consensuses.fa"
    def cleaned   = "${results}/cleaned_consensus/all_clusters_cleaned_consensus.fa"
    """
    set -euo pipefail
    mkdir -p ${mod.name}_reports

    # Pass a consensus FASTA only if the core produced one (module tolerates absence).
    cons_arg=""
    if [ -f "${cleaned}" ]; then cons_arg="--consensus-fasta ${cleaned}"
    elif [ -f "${cons}" ]; then cons_arg="--consensus-fasta ${cons}"; fi

    python \$GAMECA_HOME/${mod.script} \\
        --input       "${clustered}" \\
        --reports-dir "${mod.name}_reports" \\
        --family      "${meta.id}" \\
        ${mod.args} \$cons_arg

    cat <<-END_VERSIONS > versions.yml
    "${task.process}:${mod.name}":
        script: ${mod.script}
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${mod.name}_reports
    echo "${mod.name}" > ${mod.name}_reports/${mod.name}.stub.txt
    echo '"${task.process}:${mod.name}": {stub: true}' > versions.yml
    """
}
