# TEST_TE Analysis Results

Generated: 2026-01-28 14:46:41

## Quick Start

1. Open **visualizations/index.html** in your browser for interactive dashboard
2. Read **overall_statistics.txt** for summary statistics
3. Check **selected_primers_summary.csv** for best primers

## Dataset Summary

- Total sequences: 15
- Number of clusters: 1
- Noise sequences: 0

## Directory Structure

```
test_te/
├── README.txt                          (this file)
├── overall_statistics.txt              (complete statistics)
├── cluster_comparison.txt              (cluster summary)
├── test_te_clustered.csv (data with clusters)
├── test_te_seqs.fa       (all sequences)
├── test_te_aligned.fa    (alignment)
├── test_te_consensus.fa  (consensus)
├── selected_primers_summary.csv        (best primers)
├── visualizations/                     (interactive plots)
│   └── index.html                      (🎯 START HERE)
├── cialign_plots/                      (alignment visualizations)
│   └── index.html                      (🎨 ALIGNMENT PLOTS)
├── cleaned_consensus/                  (CIAlign cleaned consensus)
│   ├── {family}_cleaned_consensus.fa
│   ├── cluster_*_cleaned_consensus.fa
│   └── all_clusters_cleaned_consensus.fa
├── cluster_statistics/                 (per-cluster stats)
└── cluster_alignments/                 (per-cluster alignments)
```

## Key Output Files

### Statistics
- **overall_statistics.txt** - Length, GC, expression stats for all sequences
- **cluster_comparison.txt** - Quick comparison across clusters
- **cluster_statistics/cluster_{n}_statistics.txt** - Stats for each cluster

### Sequences & Alignments
- **{family}_seqs.fa** - All sequences in FASTA format
- **{family}_aligned.fa** - Multiple sequence alignment (MAFFT)
- **{family}_consensus.fa** - Consensus sequence from alignment
- **cluster_alignments/** - Per-cluster alignments and consensuses
- **cleaned_consensus/** - CIAlign cleaned consensus sequences

### Primers
- **selected_primers_summary.csv** - Top 8 primers (global)
- **cluster_top5_primers.csv** - Top 5 primers per cluster
- **{primer}_genome_hits.csv** - Genome locations for each primer
- **primer_genome_hits_summary.csv** - Summary of genome hits

### Visualizations
- **visualizations/index.html** - Interactive dashboard
- **cialign_plots/index.html** - CIAlign alignment visualizations
- **clustering_visualization.html** - PCA/UMAP/t-SNE plots
- **visualizations/cluster_distribution.html** - Cluster sizes
- **visualizations/sequence_characteristics.html** - Length/GC plots
- **visualizations/expression_heatmap.html** - Expression by cluster
- **visualizations/expression_comparison.html** - Condition comparison

## Understanding Your Results

### Clusters
Sequences were grouped by sequence similarity using:
- 12-mer frequency analysis
- UMAP dimensionality reduction  
- HDBSCAN clustering

Cluster -1 = noise/outliers that don't fit into any cluster.

### Primers
Two optimization strategies were used:
1. **cov_then_expr**: Maximize coverage first, then expression
2. **expr_then_cov**: Maximize expression first, then coverage

Best primers typically rank high in both strategies.

### Expression (if available)
Expression data across conditions shows:
- Which sequences are highly expressed
- Differential expression between conditions
- Cluster-specific expression patterns

## Common Next Steps

1. **Experimental Design**
   - Check selected_primers_summary.csv
   - Verify genome specificity in primer_genome_hits_summary.csv
   - Design PCR/qPCR experiments

2. **Sequence Analysis**
   - View alignments in Jalview/AliView
   - BLAST consensus sequences
   - Compare cluster consensuses

3. **Publication**
   - Export plots from visualizations/
   - Copy statistics from .txt files
   - Include primer tables

## Parameters Used

- K-mer size: 18
- Top primers (global): 8
- Top primers (per cluster): 5
- Genome reference: /project/amodzlab/index/human/hg38/hg38.fa

## Tools Compatibility

These files work with:
- **Alignment Viewers**: Jalview, AliView, Geneious
- **BLAST**: Use consensus.fa at NCBI
- **Primer Design**: Primer3, OligoAnalyzer
- **Statistics**: R, Python pandas, Excel
- **Genome Browsers**: UCSC, IGV

## Questions or Issues?

- Check OUTPUT_GUIDE.md for detailed documentation
- Verify input data format
- Ensure MAFFT is installed for alignments

Analysis pipeline version: 1.0
