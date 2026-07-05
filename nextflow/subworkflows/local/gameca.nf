// GAMECA — the composable end-to-end subworkflow.
//
// Include this from any other nf-core-style pipeline:
//   include { GAMECA } from './subworkflows/local/gameca.nf'
//   workflow { GAMECA( ch_families ) }         // ch: tuple(meta, path(input_csv|[]))
//
// meta = [ id: <family>, assembly: <hg38|mm10|...> ]
//
// DAG: every pipeline stage is its own Nextflow process (no fused "core" step).
//   GENOME -> DATA -> SEQUENCES -> STATS -> CLUSTERING
//     -> { DASHBOARD, ALIGNMENT, MOTIF, PRIMERS }  (parallel fan-out)
//     -> MERGE  (folds the four branches back into one family folder)
//     -> STANDOUT  (15-module parallel fan-out, see standout.nf)
//     -> REPORT
// GENOME and SEQUENCES/PRIMERS reload the genome pickle GENOME wrote to the
// family folder instead of re-parsing the FASTA (see te_genome.GenomeCache).

include { GAMECA_GENOME     } from '../../modules/local/genome.nf'
include { GAMECA_DATA       } from '../../modules/local/data.nf'
include { GAMECA_SEQUENCES  } from '../../modules/local/sequences.nf'
include { GAMECA_STATS      } from '../../modules/local/stats.nf'
include { GAMECA_CLUSTERING } from '../../modules/local/clustering.nf'
include { GAMECA_DASHBOARD  } from '../../modules/local/dashboard.nf'
include { GAMECA_ALIGNMENT  } from '../../modules/local/alignment.nf'
include { GAMECA_MOTIF      } from '../../modules/local/motif.nf'
include { GAMECA_PRIMERS    } from '../../modules/local/primers.nf'
include { GAMECA_MERGE      } from '../../modules/local/merge.nf'
include { STANDOUT          } from './standout.nf'

// Nextflow's default .join() keys on the tuple's first element; joining
// directly on the `meta` Map works but re-keying by the plain meta.id string
// (same idiom standout.nf already uses) is the safer, well-tested pattern for
// multi-way joins, so every join below goes through this helper.
def keyed(ch) { ch.map { meta, x -> tuple(meta.id, meta, x) } }

workflow GAMECA {
    take:
    ch_input            // tuple val(meta), path(input_csv)  (input_csv may be an empty list)

    main:
    ch_versions = Channel.empty()

    GAMECA_GENOME( ch_input )
    ch_versions = ch_versions.mix( GAMECA_GENOME.out.versions )

    ch_data_in = keyed( GAMECA_GENOME.out.results )
        .join( keyed( ch_input ) )
        .map { id, meta, results, meta2, input_csv -> tuple( meta, results, input_csv ) }
    GAMECA_DATA( ch_data_in )
    ch_versions = ch_versions.mix( GAMECA_DATA.out.versions )

    GAMECA_SEQUENCES( GAMECA_DATA.out.results )
    ch_versions = ch_versions.mix( GAMECA_SEQUENCES.out.versions )

    GAMECA_STATS( GAMECA_SEQUENCES.out.results )
    ch_versions = ch_versions.mix( GAMECA_STATS.out.versions )

    GAMECA_CLUSTERING( GAMECA_STATS.out.results )
    ch_versions = ch_versions.mix( GAMECA_CLUSTERING.out.versions )

    // Fan out: dashboard/alignment/motif/primers only need the clustered CSV,
    // so they all run in parallel off the same CLUSTERING output.
    GAMECA_DASHBOARD( GAMECA_CLUSTERING.out.results )
    GAMECA_ALIGNMENT( GAMECA_CLUSTERING.out.results )
    GAMECA_MOTIF(     GAMECA_CLUSTERING.out.results )
    GAMECA_PRIMERS(   GAMECA_CLUSTERING.out.results )
    ch_versions = ch_versions.mix(
        GAMECA_DASHBOARD.out.versions, GAMECA_ALIGNMENT.out.versions,
        GAMECA_MOTIF.out.versions,     GAMECA_PRIMERS.out.versions
    )

    ch_merge_in = keyed( GAMECA_CLUSTERING.out.results )
        .join( keyed( GAMECA_DASHBOARD.out.results ) )
        .join( keyed( GAMECA_ALIGNMENT.out.results ) )
        .join( keyed( GAMECA_MOTIF.out.results ) )
        .join( keyed( GAMECA_PRIMERS.out.results ) )
        .map { id, meta, base, meta2, dash, meta3, align, meta4, motif, meta5, primers ->
            tuple( meta, base, dash, align, motif, primers )
        }

    GAMECA_MERGE( ch_merge_in )

    if ( params.skip_standout ) {
        ch_results = GAMECA_MERGE.out.results
        ch_reports = Channel.empty()
    } else {
        STANDOUT( GAMECA_MERGE.out.results )
        ch_versions = ch_versions.mix( STANDOUT.out.versions )
        ch_results  = STANDOUT.out.results
        ch_reports  = STANDOUT.out.reports
    }

    emit:
    results  = ch_results
    reports  = ch_reports
    versions = ch_versions
}
