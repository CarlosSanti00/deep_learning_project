# Module loading
import pandas as pd
import gzip

# File definitions
unlabeled_gene_expression_file = 'data/archs4_gene_expression_norm_transposed.tsv.gz'
gene_expression_file = 'data/gtex_gene_expression_norm_transposed.tsv.gz'
isoform_expression_file = 'data/gtex_isoform_expression_norm_transposed.tsv.gz'
isoform_annotation = 'data/gtex_gene_isoform_annoation.tsv.gz'
tissue_annotation = 'data/gtex_annot.tsv.gz'

# Gene-expression dataframe read
df_ge = pd.read_csv(gene_expression_file, compression='gzip', header=0, sep='\t', quotechar='"', error_bad_lines=False)
print(f'Shape of the labeled gene expression dataset: {df_ge.shape}\n')

reader_ge = pd.read_csv(gene_expression_file, compression='gzip', header=0, sep='\t', quotechar='"', error_bad_lines=False, chunksize=128)

# Isoform-expression dataframe read (in chunks)
reader_isoform = pd.read_csv(isoform_expression_file, compression='gzip', header=0, sep='\t', quotechar='"', error_bad_lines=False, chunksize=128)

# print('Isoform-expression dataframe preview:')
# for i, chunk in enumerate(reader_isoform):
#     if i >= 2:
#         break  # Exit the loop after two iterations
#     print(chunk)

total_rows = 0
total_cols = 0
for chunk_ge, chunk_isoform in zip(reader_ge, reader_isoform):

    # Check if the first column (sample IDs) of both chunks match
    if not chunk_ge.iloc[:, 0].equals(chunk_isoform.iloc[:, 0]):
        print("Error: Sample IDs in gene expression and isoform expression chunks do not match!")
        break
    
    # Process total shape information for isoform expression DataFrame
    total_rows += chunk_isoform.shape[0]
    total_cols = max(total_cols, chunk_isoform.shape[1])

print("Total shape of the isoform-expression file: Rows =", total_rows, ", Columns =", total_cols)

# Isoform annotation dataframe read
df_if_note = pd.read_csv(isoform_annotation, compression='gzip', header=0, sep='\t', quotechar='"', error_bad_lines=False)
print(f'\nShape of the isoform annotation file: {df_if_note.shape}')

# Tissue annotation dataframe read
df_tissue_note = pd.read_csv(tissue_annotation, compression='gzip', header=0, sep='\t', quotechar='"', error_bad_lines=False)

# Unique number of tissues
unique_tissues = df_tissue_note['tissue'].unique()
print(f'\nNumber of possible tissues: {len(unique_tissues)}\n')

# Print the counts for each tissue type
tissue_counts = df_tissue_note['tissue'].value_counts()
print(tissue_counts)

# # Unlabeled gene-expression dataframe

# # Open the compressed file in binary read mode
# with gzip.open(unlabeled_gene_expression_file, 'rb') as f:
#     # Read the content of the compressed file and decode it as UTF-8
#     content = f.read().decode('utf-8')

# # Use pandas to read the TSV data from the decoded content
# df_archs4 = pd.read_csv(StringIO(content), sep='\t')
# df_archs4 = pd.read_csv(unlabeled_gene_expression_file, compression='gzip', header=0, sep='\t', quotechar='"', error_bad_lines=False)
# print(f'\nShape of the labeled gene expression dataset: {df_archs4.shape}')