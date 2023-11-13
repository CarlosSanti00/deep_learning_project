# Module loading
import pandas as pd
import gzip

# File definitions
unlabeled_gene_expression_file = 'data/archs4_gene_expression_norm_transposed.tsv.gz'
gene_expression_file = 'data/gtex_gene_expression_norm_transposed.tsv.gz'
isoform_expression_file = 'data/gtex_isoform_expression_norm_transposed.tsv.gz'
isoform_annotation = 'data/gtex_gene_isoform_annoation.tsv.gz'
tissue_annotation = 'data/gtex_annot.tsv.gz'

# Unlabeled gene-expression dataframe read (in chunks)
reader_archs4 = pd.read_csv(unlabeled_gene_expression_file, compression='gzip', header=0, sep='\t', quotechar='"', error_bad_lines=False, chunksize=128)

total_rows = 0
total_cols = 0
for chunk_archs4 in reader_archs4:

    # Process total shape information for isoform expression DataFrame
    total_rows += chunk_archs4.shape[0]
    total_cols = max(total_cols, chunk_archs4.shape[1])

print("Total shape of the unlabeled gene-expression file: Rows =", total_rows, ", Columns =", total_cols)