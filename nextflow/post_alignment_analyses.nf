#!/usr/bin/env nextflow
// ─────────────────────────────────────────────────────────────────────────────
// GAMECA post-alignment analyses entrypoint. Runs only the standout-analysis modules (in
// parallel) + report against an ALREADY-BUILT family folder.
//
// This is what query.py invokes when run with --post-alignment-analyses-nextflow: query.py does
// Stages 1-10 in-process, then hands the finished folder to Nextflow so the 15
// post-alignment analyses modules run concurrently instead of in query.py's sequential loop.
//
//   nextflow run nextflow/post_alignment_analyses.nf --results results/hervk9 \
//       --family HERVK9 --assembly hg38 -profile lsf,singularity
// ─────────────────────────────────────────────────────────────────────────────

nextflow.enable.dsl = 2

include { STANDOUT } from './subworkflows/local/standout.nf'

workflow {
    if ( !params.results ) error "Provide --results <existing GAMECA family folder>."
    if ( !params.family  ) error "Provide --family <NAME> (must match the folder's clustered CSV)."

    def meta = [ id: params.family, assembly: params.assembly ]
    ch_results = Channel.of( tuple( meta, file( params.results, checkIfExists: true ) ) )

    STANDOUT( ch_results )

    STANDOUT.out.versions
        .unique()
        .collectFile( name: 'gameca_post_alignment_analyses_versions.yml', storeDir: params.tracedir )
}
