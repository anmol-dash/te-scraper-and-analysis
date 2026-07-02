// GAMECA — the composable end-to-end subworkflow.
//
// Include this from any other nf-core-style pipeline:
//   include { GAMECA } from './subworkflows/local/gameca.nf'
//   workflow { GAMECA( ch_families ) }         // ch: tuple(meta, path(input_csv|[]))
//
// meta = [ id: <family>, assembly: <hg38|mm10|...> ]

include { GAMECA_CORE } from '../../modules/local/core.nf'
include { STANDOUT    } from './standout.nf'

workflow GAMECA {
    take:
    ch_input            // tuple val(meta), path(input_csv)  (input_csv may be an empty list)

    main:
    ch_versions = Channel.empty()

    GAMECA_CORE( ch_input )
    ch_versions = ch_versions.mix( GAMECA_CORE.out.versions )

    if ( params.skip_standout ) {
        ch_results = GAMECA_CORE.out.results
        ch_reports = Channel.empty()
    } else {
        STANDOUT( GAMECA_CORE.out.results )
        ch_versions = ch_versions.mix( STANDOUT.out.versions )
        ch_results  = STANDOUT.out.results
        ch_reports  = STANDOUT.out.reports
    }

    emit:
    results  = ch_results
    reports  = ch_reports
    versions = ch_versions
}
