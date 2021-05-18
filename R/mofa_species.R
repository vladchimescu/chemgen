## MOFA analysis of E. coli and S. typhimurium data
library(tidyverse)
library(MultiAssayExperiment)
library(MOFA)
library(reticulate)

# aggregate chemgen data across concentrations
agg_conc <- function(df, fdr = 0.05) {
  df_avg = group_by(df, gene, drug) %>%
    summarise(hit = sum(padj < fdr),
              sscore = sscore[which.max(abs(sscore))]) %>%
    ungroup()
  df_avg
}

use_condaenv(condaenv = "/g/huber/users/vkim/conda-envs/mofaenv",
             conda = '/g/funcgen/gbcs/miniconda2/bin/conda')

get_gramneg_pval <- function(datadir) {
  eps_all_df = readRDS(file.path(datadir, "eps-gramneg-quart.rds"))
  eps_all_df = dplyr::mutate(eps_all_df, efsize = ifelse(abs(eps.q1) > abs(eps.q3),
                                                         eps.q1, eps.q3))
  
  pval_ana = read.table(file.path(datadir, "p-values-gramneg.txt"),
                        sep="\t", header = T, stringsAsFactors = F)
  pval_ana = dplyr::rename(pval_ana, Strain = bug,
                           pval = CompleteConcRange,
                           comb = combination)
  pval_ana = group_by(pval_ana, Strain) %>% 
    mutate(padj = p.adjust(pval, method = "BH"))
  
  pval_ana$Strain = plyr::mapvalues(pval_ana$Strain, 
                                    from = c("Ecoli_BW", "Ecoli_iAi1",
                                             "PA_01", "PA_14", "ST_14028", "ST_LT2"),
                                    to = c("E. coli BW", "E. coli iAi1",
                                           "PAO1", "PA14",
                                           "ST 14028s", "ST LT2"))
  pval_ana = inner_join(pval_ana, eps_all_df)
  pval_ana = mutate(pval_ana,
                    d1 = gsub("(.+)_(.+)", "\\1", comb),
                    d2 = gsub("(.+)_(.+)", "\\2", comb))
  pval_ana$comb = purrr::map2_chr(as.character(pval_ana$d1),
                                  as.character(pval_ana$d2),
                                  ~ paste(sort(c(.x, .y)), collapse = "_"))
  pval_ana
}

#basedir = "~/Documents/github/chemgen"
basedir = '.'
datadir = file.path(basedir, 'data/shiny')
figdir = file.path(basedir, "figures/")

# color palette
pal = c("#FFCC33", "#009999")

# load chemical genetics data
ecoli_chemgen = agg_conc(readRDS(file.path(datadir, 'nichols.rds')))
st_chemgen = agg_conc(readRDS(file.path(datadir, 'ST_chemgen.rds')))

# map E. coli gene names
string.map = read.csv(file.path(datadir, "string_mapping-EcoliBW.tsv"),
                      header = T,
                      sep = "\t", stringsAsFactors = F)
ecoli_chemgen = mutate(ecoli_chemgen,
                       gene = plyr::mapvalues(gene, from = string.map$queryItem,
                                              to = string.map$preferredName))

# load drug-drug interaction data
bliss_all = get_gramneg_pval(datadir = file.path(basedir, "data/strains"))
ecoli_df = filter(bliss_all, Strain == "E. coli BW")
ST_df = filter(bliss_all, Strain == 'ST 14028s')

# subset interaction profiles only to drugs present in chemgen data
drugs_chemgen = union(st_chemgen$drug, ecoli_chemgen$drug)
ecoli_df = filter(ecoli_df, d1 %in% drugs_chemgen & d2 %in% drugs_chemgen)
ST_df = filter(ST_df, d1 %in% drugs_chemgen & d2 %in% drugs_chemgen)



# convert all data.frames/tibbles into matrices

chemgen_tomat <- function(df) {
  df_wide = tidyr::spread(select(df, drug, gene, sscore),
                          gene, sscore)
  
  mat = as.matrix(df_wide[,-1])
  rownames(mat) = df_wide$drug
  t(mat)
}

ec_cg_mat = chemgen_tomat(ecoli_chemgen)
st_cg_mat = chemgen_tomat(st_chemgen)

bliss_tomat <- function(df) {
  df = select(ungroup(df), d1, d2, efsize)
  df_wide = bind_rows(df,
                      dplyr::rename(df, 
                                    d1 = d2,
                                    d2 = d1)) %>%
    tidyr::spread(d1, efsize)
  
  mat = as.matrix(df_wide[,-1])
  rownames(mat) = df_wide$d2
  
  t(mat)
}

ec_bliss_mat = bliss_tomat(ecoli_df)
st_bliss_mat = bliss_tomat(ST_df)

# create a MultiAssayExperiment object
objlist = list("EC drug interactions" = ec_bliss_mat,
               "ST drug interactions" = st_bliss_mat,
               "EC chemical genetics" = ec_cg_mat,
               "ST chemical genetics" = st_cg_mat)

mae = MultiAssayExperiment(objlist)

MOFA_obj <- createMOFAobject(mae) 
data_overview <- plotDataOverview(MOFA_obj)

print(data_overview)



# set model parameter
TrainOptions <- getDefaultTrainOptions()
ModelOptions <- getDefaultModelOptions(MOFA_obj)
DataOptions <- getDefaultDataOptions()

# number of LF's
#num_factors = 10
# ModelOptions$numFactors <- num_factors

# at least 5% variance explained
TrainOptions$DropFactorThreshold <- 0.05 
# % variance explained by factor

TrainOptions$tolerance <- 0.01 # 0.01 is recommended
TrainOptions$seed <- 1705 #random number
DataOptions$scaleViews <- TRUE #unit variance, if scale is too different
#TrainOptions$maxiter <- 10 # only for testing

MOFAobject <- prepareMOFA(
  MOFA_obj,
  DataOptions = DataOptions,
  ModelOptions = ModelOptions,
  TrainOptions = TrainOptions)

# use 10 random initializations
n_inits <- 10
model_output_path = file.path(basedir, 'data/mofa_out')
model_name = 'chemgenbliss'

# runMOFA(MOFAobject,
#        outfile = paste0(model_output_path,
#                         "/", Sys.Date(),
#                         "_", model_name,
#                         ".hdf5"))


model_list <- lapply(seq_len(n_inits), function(it) {
  TrainOptions$seed <- 1705 + it

  MOFAobject <- prepareMOFA(
    MOFA_obj,
    DataOptions = DataOptions,
    ModelOptions = ModelOptions,
    TrainOptions = TrainOptions)
  runMOFA(MOFAobject,
          outfile = paste0(model_output_path,
                           "/", Sys.Date(),
                           "_", model_name,"_",
                           it,
                           ".hdf5"))
})

save(model_list,
     file = paste0(model_output_path, "/",model_name,"_list.RData"))

