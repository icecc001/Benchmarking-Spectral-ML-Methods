## BIOS774 HW1 Benchmarking Spectral Methods (Data Two)
# Author: Xinyu Zhang
# Date: Sept. 30th

# Import Packages
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")

results_file = "hw1/data/write/pbmc3k.h5ad"  # the file that will store the analysis results

#################################
## 0. Data Preprocessing
#################################

adata = sc.read_10x_mtx(
    "hw1/data/filtered_gene_bc_matrices/hg19/",  # the directory with the `.mtx` file
    var_names="gene_symbols",  # use gene symbols for the variable names (variables-axis index)
    cache=True,  # write a cache file for faster subsequent reading
)

adata.var_names_make_unique()
adata

## basic filtering
## filtered out 19024 genes that are detected in less than 3 cells

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# annotate the group of mitochondrial genes as "mt"
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)

'''
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)


sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt")
sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")
'''

## filtering by slicing the AnnData object.
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

# Total-count normalize (library-size correct) the data matrix 
# to 10,000 reads per cell, so that counts become comparable among cells.
sc.pp.normalize_total(adata, target_sum=1e4)

# Logarithmize the data:
sc.pp.log1p(adata)

# highly variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

adata = adata[:, adata.var.highly_variable]

sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])

# Scale each gene to unit variance. Clip values exceeding standard deviation 10.
sc.pp.scale(adata, max_value=10)

## PCA intake
sc.tl.pca(adata, svd_solver="arpack")


sc.pl.pca(adata, color="CST3", save = "pca-inside.png")

if "CST3" in adata.var_names:
    # Extract the expression values for "CST3"
    cst3_expression = adata[:, "CST3"].X
    
    # If X is a sparse matrix, convert it to a dense array
    if hasattr(cst3_expression, "toarray"):
        cst3_expression = cst3_expression.toarray()
    
    # Convert the expression data to a 1D array
    cst3_expression = cst3_expression.flatten()
    
    # Display the CST3 expression values
    print(cst3_expression)

## in some paper
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

sc.tl.leiden(
    adata,
    resolution=0.9,
    random_state=0,
    flavor="igraph",
    n_iterations=2,
    directed=False,
)

sc.pl.pca(adata, color="leiden", save = "pca-inside-leiden.png")


## n_obs = 2638, n_vars = 1838
X = adata.X  # Extract the matrix (observations × variables)

# If X is a sparse matrix (common in AnnData), convert it to a dense array
if hasattr(X, "toarray"):
    X = X.toarray()

df = pd.DataFrame(X)

# Set the column names to var_names (gene names) and the row names to obs_names (observation names)
df.columns = adata.var_names  # Set column names (genes)
df.index = adata.obs_names    # Set row names (observations)

# Display the resulting DataFrame
print(df.head())

#leiden_clusters = adata.obs['leiden']
#df['leiden_clusters'] = leiden_clusters.values

df.to_csv('hw1/data/3kPBMC.csv', index=False)