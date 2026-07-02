#!/usr/bin/env nextflow
// ─────────────────────────────────────────────────────────────────────────────
// GAMECA — top-level entrypoint. Runs the full pipeline (Stages 1-10 + post-alignment analyses + report)
// for every family in a samplesheet, in parallel, on Nextflow.
//
//   nextflow run nextflow/main.nf --input samplesheet.csv -profile lsf,singularity
//   nextflow run nextflow/main.nf --family HERVK9 --assembly hg38 -profile lsf,singularity
//   nextflow run nextflow/main.nf -profile test -stub-run
//
// samplesheet.csv columns:  family,assembly,input
//   family    TE repName (required)
//   assembly  hg38 | mm10 | ...   (optional; defaults to --assembly)
//   input     path to a prebuilt loci CSV  (optional; blank → pipeline fetches)
// ─────────────────────────────────────────────────────────────────────────────

nextflow.enable.dsl = 2

include { GAMECA } from './subworkflows/local/gameca.nf'

// Build the (meta, input) channel from either --input samplesheet or --family.
def parseInput() {
    if ( params.input ) {
        return Channel
            .fromPath( params.input, checkIfExists: true )
            .splitCsv( header: true )
            .map { row ->
                if ( !row.family?.trim() ) error "samplesheet row missing 'family': ${row}"
                def meta = [ id: row.family.trim(),
                             assembly: (row.assembly?.trim() ?: params.assembly) ]
                def inp  = (row.input?.trim())
                             ? file( row.input.trim(), checkIfExists: true )
                             : []
                tuple( meta, inp )
            }
    }
    if ( params.family ) {
        def meta = [ id: params.family, assembly: params.assembly ]
        def inp  = params.input_csv ? file( params.input_csv, checkIfExists: true ) : []
        return Channel.of( tuple( meta, inp ) )
    }
    error "Provide either --input <samplesheet.csv> or --family <NAME>."
}

workflow {
    GAMECA( parseInput() )

    GAMECA.out.versions
        .unique()
        .collectFile( name: 'gameca_versions.yml', storeDir: params.tracedir )
}
