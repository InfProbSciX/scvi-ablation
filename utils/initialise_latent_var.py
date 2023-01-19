
import math
import numpy as np


cc_genes = {
    's_genes': ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", 
"RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", 
"MLF1IP", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", 
"WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", 
"RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", 
"USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"], 
    'g2m_genes': ["HMGB2", 
"CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", 
"CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", 
"SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", 
"ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", 
"CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", 
"CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", 
"ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", 
"CBX5", "CENPA"]}


def get_CC_effect_init(adata, cc_genes):
    
    mean_s_gene_expr = np.asarray(adata[:, np.intersect1d(adata.var_names, cc_genes['s_genes']) ].X.mean(axis=1)).flatten()  
    mean_g2m_gene_expr = np.asarray(adata[:, np.intersect1d(adata.var_names, cc_genes['g2m_genes']) ].X.mean(axis=1)).flatten()  
    
    X_cc = np.transpose(np.asarray([mean_s_gene_expr, mean_g2m_gene_expr])) 

    #  scaling 2d cell vector to unit vector
    scaling_factor = np.sqrt(np.sum(np.power(X_cc,2), axis=1)).reshape(-1,1)  
    scaling_factor[np.where(scaling_factor.flatten() == 0)]  = 1 # to avoid zero_divs | # if both S and G2M mean expr is 0, then cc is 0
    X = np.divide(X_cc, scaling_factor)

    # take only the 2nd column -G2M (as after scaling, each column is informative of the other)
    # compute the sin inverse (Note: cc_gene_expr = sin(cc_effect) => cc_effect = asin(cc_gene_expr) ) 
    asin = np.vectorize(math.asin); cc = asin(X[:,1])
    cc[X[:,0]<0]=np.pi-cc[X[:,0]<0] # not yet convinced whether this condition is ever triggered in log1p expression data
    cc = (cc+np.pi) % (2*np.pi) # mod 2pi

    return cc
