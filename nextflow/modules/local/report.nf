// GAMECA report — merge every post-alignment analyses module's isolated reports back into the
// family folder, then render the HTML (and optionally LaTeX/PDF) report.
//
// generate_report.py / generate_latex_report.py both take a family folder as
// their argument (see their module docstrings).
//
// Input: tuple val(meta), path(results), path(post_alignment_analyses_reports, stageAs 'post_alignment_analyses/*')

process GAMECA_REPORT {
    tag   { meta.id }
    label 'process_low'

    input:
    tuple val(meta), path(results), path(post_alignment_analyses_reports, stageAs: 'post_alignment_analyses/*')

    output:
    tuple val(meta), path("${results}")           , emit: results
    tuple val(meta), path("${results}/reports")   , emit: reports
    path "versions.yml"                           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def pdf = params.report_pdf ? "python \$GAMECA_HOME/generate_latex_report.py ${results} || true" : ''
    """
    set -euo pipefail
    mkdir -p ${results}/reports

    # Fold each module's isolated <name>_reports/ back into the family reports/.
    for d in post_alignment_analyses/*; do
        [ -d "\$d" ] || continue
        cp -a "\$d"/. ${results}/reports/ 2>/dev/null || true
    done

    ${params.report_html ? "python \$GAMECA_HOME/generate_report.py ${results}" : 'true'}
    ${pdf}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gameca: \${GAMECA_VERSION:-0.4.42}
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${results}/reports
    for d in post_alignment_analyses/*; do [ -d "\$d" ] && cp -a "\$d"/. ${results}/reports/ 2>/dev/null || true; done
    echo "<html>stub</html>" > ${results}/reports/report.html
    echo '"${task.process}": {stub: true}' > versions.yml
    """
}
