// STANDOUT — post-alignment analyses orchestration. Builds the module registry (mirroring
// query.py `_MODULES`), scatters each module across GAMECA_STANDOUT so they run
// in parallel, then collects their outputs into GAMECA_REPORT.
//
// This is the single source of truth for the post-alignment analyses registry on the Nextflow
// side and is reused by both main.nf (full pipeline) and post_alignment_analyses.nf (the entry
// query.py calls when --post-alignment-analyses-nextflow is set).

include { GAMECA_STANDOUT } from '../../modules/local/standout.nf'
include { GAMECA_REPORT   } from '../../modules/local/report.nf'

// post-alignment analyses registry. Optional knobs fall through to each module's own default
// (only emitted when the matching param is set), exactly like query.py does.
def opt(flag, val)      { val != null ? "${flag} ${val}" : '' }
def optList(flag, vals) {
    if ( !vals ) return ''
    def items = (vals instanceof CharSequence) ? vals.toString().split(/[,\s]+/) as List : (vals as List)
    items = items.findAll { it }
    items ? "${flag} ${items.join(' ')}" : ''
}

def buildRegistry(meta) {
    def asm = meta.assembly ?: params.assembly
    [
        [ name: 'phylo',         script: 'run_phylo_analysis.py',
          args: "--assembly ${asm} --clock-divisor ${params.clock_divisor} --intact-orf-aa ${params.intact_orf_aa} " +
                opt('--subst-rate', params.subst_rate) + ' ' + opt('--mafft-cmd', params.mafft_cmd) ],
        [ name: 'grna',          script: 'run_grna_offtarget.py',
          args: "--cas ${params.grna_cas} --max-mm ${params.grna_max_mm} " + opt('--background', params.grna_background) ],
        [ name: 'transduction',  script: 'run_transduction.py',
          args: "--tail-bp ${params.tail_bp} --min-shared 3" ],
        [ name: 'antisense',     script: 'run_antisense_promoter.py',
          args: "--promoter-bp ${params.promoter_bp}" ],
        [ name: 'ctcf_tad',      script: 'run_ctcf_tad.py',
          args: "--motif-mismatch 3 " + opt('--ctcf-preset', params.ctcf_preset) + ' ' + opt('--tads-preset', params.tads_preset) ],
        [ name: 'epigenetic',    script: 'run_epigenetic_overlay.py',
          args: opt('--preset', params.epigenetic_preset) ],
        [ name: 'ortholog',      script: 'run_ortholog_insertion.py',
          args: optList('--species', params.ortholog_species) + ' ' + opt('--liftover-cmd', params.liftover_cmd) ],
        [ name: 'multiassembly', script: 'run_multiassembly_liftover.py',
          args: "--source-assembly ${asm} " + optList('--target-assemblies', params.target_assemblies) + ' ' + opt('--liftover-cmd', params.liftover_cmd) ],
        [ name: 'fold',          script: 'run_fold_prediction.py',
          args: "--per-cluster --min-aa 100 --top-n 5 " + opt('--colabfold-cmd', params.colabfold_cmd) ],
        [ name: 'divergence',    script: 'run_divergence.py',
          args: "--assembly ${asm} " + opt('--cpg-omega', params.cpg_omega) ],
        [ name: 'ltr_struct',    script: 'run_ltr_struct.py',
          args: opt('--min-ltr-identity', params.min_ltr_identity) ],
        [ name: 'subfamily',     script: 'run_subfamily.py',
          args: "--assembly ${asm} " + opt('--cpg-omega', params.cpg_omega) ],
        [ name: 'benchmark',     script: 'run_benchmark.py',
          args: "--assembly ${asm}" ],
        [ name: 'motif_gain',    script: 'run_motif_gain.py',
          args: "--assembly ${asm}" ],
        [ name: 'expression',    script: 'te_expression.py',
          args: "--out-dir ." ],
    ]
}

workflow STANDOUT {
    take:
    ch_results          // tuple val(meta), path(results)

    main:
    ch_versions = Channel.empty()

    // One (meta, results, mod) per post-alignment analyses module → parallel fan-out.
    // params.skip_modules (comma/space-separated names) mirrors query.py/
    // run_stage11_all.py's --skip so both engines can exclude the same modules
    // (e.g. ColabFold, which is heavy/GPU and off by default in test harnesses).
    def skipMods = (params.skip_modules ?: '').split(/[,\s]+/).findAll { it } as Set
    ch_mods = ch_results.flatMap { meta, results ->
        buildRegistry(meta).findAll { !skipMods.contains(it.name) }
                            .collect { mod -> tuple(meta, results, mod) }
    }

    GAMECA_STANDOUT( ch_mods )
    ch_versions = ch_versions.mix( GAMECA_STANDOUT.out.versions )

    // Gather every module's reports back per family, rejoin with its folder.
    ch_collected = GAMECA_STANDOUT.out.reports
        .map { meta, rep -> tuple( meta.id, meta, rep ) }
        .groupTuple( by: 0 )
        .map { id, metas, reps -> tuple( metas[0], reps ) }

    ch_report_in = ch_results
        .map { meta, results -> tuple( meta.id, meta, results ) }
        .join( ch_collected.map { meta, reps -> tuple( meta.id, reps ) } )
        .map { id, meta, results, reps -> tuple( meta, results, reps ) }

    GAMECA_REPORT( ch_report_in )
    ch_versions = ch_versions.mix( GAMECA_REPORT.out.versions )

    emit:
    results  = GAMECA_REPORT.out.results
    reports  = GAMECA_REPORT.out.reports
    versions = ch_versions
}
