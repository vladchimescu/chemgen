## MOFA analysis of E. coli and S. typhimurium data
library(tidyverse)
library(MultiAssayExperiment)
library(MOFA)
library(reticulate)
library(reshape2)
library(cowplot)

# aggregate chemgen data across concentrations
agg_conc <- function(df, fdr = 0.05) {
  df_avg = group_by(df, gene, drug) %>%
    summarise(hit = sum(padj < fdr),
              sscore = sscore[which.max(abs(sscore))]) %>%
    ungroup()
  df_avg
}

plotDataOverview2 <- function (object, colors = NULL, fontsize =9) {
  if (!is(object, "MOFAmodel")) 
    stop("'object' has to be an instance of MOFAmodel")
  TrainData <- getTrainData(object)
  M <- getDimensions(object)[["M"]]
  N <- getDimensions(object)[["N"]]
  if (is.null(colors)) {
    palette <- c("#D95F02", "#377EB8", "#E6AB02", "#31A354", 
                 "#7570B3", "#E7298A", "#66A61E", "#A6761D", "#666666", 
                 "#E41A1C", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", 
                 "#A65628", "#F781BF", "#1B9E77")
    if (M < 17) 
      colors <- palette[seq_len(M)]
    else colors <- rainbow(M)
  }
  if (length(colors) != M) 
    stop("Length of 'colors' does not match the number of views")
  names(colors) <- sort(viewNames(object))
  ovw <- vapply(TrainData, function(dat) apply(dat, 2, function(s) !all(is.na(s))), 
                logical(N))
  ovw <- ovw[apply(ovw, 1, any), , drop = FALSE]
  molten_ovw <- melt(ovw, varnames = c("sample", "view"))
  molten_ovw$sample <- factor(molten_ovw$sample, levels = rownames(ovw)[order(rowSums(ovw), 
                                                                              decreasing = TRUE)])
  n <- length(unique(molten_ovw$sample))
  molten_ovw$combi <- ifelse(molten_ovw$value, as.character(molten_ovw$view), 
                             "missing")
  molten_ovw$ntotal <- paste("n=", colSums(ovw)[as.character(molten_ovw$view)], 
                             sep = "")
  molten_ovw$ptotal <- paste("d=", vapply(TrainData, nrow, 
                                          numeric(1))[as.character(molten_ovw$view)], sep = "")
  molten_ovw$view_label = paste(molten_ovw$view, molten_ovw$ptotal, 
                                sep = "\n")
  molten_ovw$label_pos <- levels(molten_ovw$sample)[n/2]
  p <- ggplot(molten_ovw, aes_string(x = "sample",
                                     y = "view_label", 
                                     fill = "combi")) + 
    geom_tile() + 
    geom_text(data = filter(molten_ovw, 
                                            sample == levels(molten_ovw$sample)[1]), 
                              aes_string(x = "label_pos", 
                                         label = "ntotal"), size = fontsize*5/14) + 
    scale_fill_manual(values = c(missing = "grey", colors)) + 
    xlab(paste0("Compounds (n=", n, ")")) + ylab("") + 
    guides(fill = FALSE) + theme(panel.background = element_rect(fill = "white"), 
                                 text = element_text(size = fontsize),
                                 axis.ticks = element_blank(), 
                                 axis.text.x = element_blank(),
                                 axis.text.y = element_text(color = "black"), 
                                 panel.grid = element_blank(),
                                 plot.margin = unit(c(5.5, 
                                                      2, 5.5, 5.5), "pt"))
  return(p)
}

plotVarianceExplained2 <- function (object, cluster = TRUE, fontsize = 9, ...) {
  R2_list <- calculateVarianceExplained(object, ...)
  fvar_m <- R2_list$R2Total
  fvar_mk <- R2_list$R2PerFactor
  fvar_mk_df <- reshape2::melt(fvar_mk, varnames = c("factor", 
                                                     "view"))
  fvar_mk_df$factor <- factor(fvar_mk_df$factor)
  if (cluster & ncol(fvar_mk) > 1) {
    hc <- hclust(dist(t(fvar_mk)))
    fvar_mk_df$view <- factor(fvar_mk_df$view, levels = colnames(fvar_mk)[hc$order])
  }
  hm <- ggplot(fvar_mk_df, aes_string(x = "view", y = "factor")) + 
    geom_tile(aes_string(fill = "value"), color = "black") + 
    guides(fill = guide_colorbar("R2")) + 
    scale_fill_gradientn(colors = c("gray97","darkblue"), guide = "colorbar") + 
    ylab("Latent factor") + 
    theme(text = element_text(size = fontsize),
          plot.title = element_text(size = fontsize, hjust = 0.5), 
          axis.title.x = element_blank(), axis.text.x = element_text(size = fontsize, 
                                                                     angle = 60, hjust = 1, vjust = 1, color = "black"), 
          #axis.text.y = element_text(size = fontsize, color = "black"), 
          axis.title.y = element_text(size = fontsize), axis.line = element_blank(), 
          axis.ticks = element_blank(), panel.background = element_blank())
  hm <- hm + ggtitle("Variance explained per factor") + guides(fill = guide_colorbar("R2"))
  fvar_m_df <- data.frame(view = factor(names(fvar_m), levels = names(fvar_m)), 
                          R2 = fvar_m)
  if (cluster == TRUE & ncol(fvar_mk) > 1) {
    fvar_m_df$view <- factor(fvar_m_df$view, levels = colnames(fvar_mk)[hc$order])
  }
  bplt <- ggplot(fvar_m_df, aes_string(x = "view", y = "R2")) + 
    ggtitle("Total variance explained per view") + geom_bar(stat = "identity", 
                                                            fill = "deepskyblue4", width = 0.9) +
    xlab("") + ylab("R2") + 
    scale_y_continuous(expand = c(0.01, 0.01)) + 
    theme(plot.margin = unit(c(1, 2.4, 0, 0), "cm"), 
          text = element_text(size = fontsize),
          panel.background = element_blank(), 
          plot.title = element_text(size = fontsize, hjust = 0.5), 
          axis.ticks.x = element_blank(), 
          axis.text.x = element_blank(),
          axis.line = element_line(size = rel(1), 
                                   color = "black"))
  p <- plot_grid(bplt, hm, align = "v", nrow = 2, rel_heights = c(1/4, 
                                                                  3/4), axis = "l")
  return(p)
}

plotFactorScatter2 <- function (object, factors, viewidx=NULL, color_by = NULL, shape_by = NULL, 
                                name_color = "", name_shape = "", dot_size = 1.5, dot_alpha = 1, 
                                showMissing = TRUE, fontsize = 9) {
  if (!is(object, "MOFAmodel")) {
    stop("'object' has to be an instance of MOFAmodel")
  }
  stopifnot(length(factors) == 2)
  if (is.numeric(factors)) {
    factors <- factorNames(object)[factors]
  }
  else {
    if (paste0(factors, collapse = "") == "all") {
      factors <- factorNames(object)
    }
    else {
      stopifnot(all(factors %in% factorNames(object)))
    }
  }
  Z <- getFactors(object, factors = factors)
  factors <- colnames(Z)
  samples <- sampleNames(object)
  N <- getDimensions(object)[["N"]]
  colorLegend <- TRUE
  if (!is.null(color_by)) {
    if (length(color_by) == 1 & is.character(color_by)) {
      if (name_color == "") 
        name_color <- color_by
      TrainData <- getTrainData(object)
      featureNames <- lapply(TrainData, rownames)
      if (color_by %in% Reduce(union, featureNames)) {
        color_by <- TrainData[[viewidx]][color_by, ]
      }
      else if (is(InputData(object), "MultiAssayExperiment")) {
        color_by <- getCovariates(object, color_by)
      }
      else {
        stop("'color_by' was specified but it was not recognised, please read the documentation")
      }
    }
    else if (length(color_by) > 1) {
      stopifnot(length(color_by) == N)
    }
    else {
      stop("'color_by' was specified but it was not recognised, please read the documentation")
    }
  }
  else {
    color_by <- rep(TRUE, N)
    names(color_by) <- sampleNames(object)
    colorLegend <- FALSE
  }
  shapeLegend <- TRUE
  if (!is.null(shape_by)) {
    if (length(shape_by) == 1 & is.character(shape_by)) {
      if (name_shape == "") 
        name_shape <- shape_by
      TrainData <- getTrainData(object)
      featureNames <- lapply(TrainData, rownames)
      if (shape_by %in% Reduce(union, featureNames)) {
        shape_by <- TrainData[[viewidx]][shape_by, ]
      }
      else if (is(InputData(object), "MultiAssayExperiment")) {
        shape_by <- getCovariates(object, shape_by)
      }
      else stop("'shape_by' was specified but it was not recognised, please read the documentation")
    }
    else if (length(shape_by) > 1) {
      stopifnot(length(shape_by) == N)
    }
    else {
      stop("'shape_by' was specified but it was not recognised, please read the documentation")
    }
  }
  else {
    shape_by <- rep(TRUE, N)
    names(shape_by) <- sampleNames(object)
    shapeLegend <- FALSE
  }
  df = data.frame(x = Z[, factors[1]], y = Z[, factors[2]], 
                  shape_by = shape_by, color_by = color_by)
  if (!showMissing) {
    df <- df[!(is.na(df$shape_by) | is.na(df$color_by)), 
    ]
  }
  if (any(is.na(df$shape_by))) {
    df$shape_by <- factor(df$shape_by, levels = c(as.character(unique(df$shape_by)), 
                                                  "NA"))
    df$shape_by[is.na(df$shape_by)] <- "NA"
  }
  else {
    df$shape_by <- as.factor(df$shape_by)
  }
  if (length(unique(df$color_by)) < 5) 
    df$color_by <- as.factor(df$color_by)
  df <- df[!(is.na(df$x) | is.na(df$y)), ]
  xlabel <- factors[1]
  ylabel <- factors[2]
  
  # add compound names
  df$drug = rownames(df)
  
  p <- ggplot(df, aes_string(x = "x", y = "y")) + 
    geom_point(aes_string(color = "color_by", 
                          shape = "shape_by"),
               size = dot_size, alpha = dot_alpha) + 
    xlab(xlabel) + ylab(ylabel) + 
    theme(text = element_text(size = fontsize),
          axis.line = element_line(size = 0.5),
          panel.border = element_blank(), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          legend.key = element_rect(fill = "white"))
  if (colorLegend) {
    p <- p + labs(color = name_color)
  }
  else {
    p <- p + guides(color = FALSE)
  }
  if (shapeLegend) {
    p <- p + labs(shape = name_shape)
  }
  else {
    p <- p + guides(shape = FALSE)
  }
  return(p)
}

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

use_condaenv(condaenv = "/g/huber/users/vkim/conda-envs/mofaenv",
             conda = '/g/funcgen/gbcs/miniconda2/bin/conda')

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
data_overview <- plotDataOverview2(MOFA_obj)
ggsave(data_overview, filename = paste0(figdir, 'MOFA-omics-overview.pdf'),
       width = 6, height = 3.5, units = 'cm')

# set model parameter
TrainOptions <- getDefaultTrainOptions()
ModelOptions <- getDefaultModelOptions(MOFA_obj)
DataOptions <- getDefaultDataOptions()

# number of LF's
#num_factors = 10
# ModelOptions$numFactors <- num_factors

# at least 5% variance explained
TrainOptions$DropFactorThreshold <- 0.01
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

#--------After saving the model you can load it--------------
# load the model
load(paste0(model_output_path, "/",model_name,"_list.RData"))
model <- selectModel(model_list)

p = plotVarianceExplained2(model)
ggsave(p, filename = paste0(figdir, 'mofa-varexplained-species.pdf'),
       width = 7, height = 11, units = 'cm')


# plot drugs in LF space
cmin = min(TrainData(model)[[1]]['ompA',], na.rm=T)
cmax = max(TrainData(model)[[1]]['ompA',], na.rm=T)
p = plotFactorScatter2(
  model,
  factors = 1:2,
  viewidx = 1,
  color_by = 'ompA',
  showMissing = F
) +   ggrepel::geom_text_repel(aes(label = ifelse((x < -0.1 & color_by < 0) | 
                                                    (color_by > 2 & x > 0.25), drug, ""),
                                   color = color_by),
                               size = 7*5/14) +
  scale_color_gradientn(colours = c('#4878d0', 'white', '#ee854a'), 
                        values = scales::rescale(c(cmin,0,cmax)),
                        guide = "colorbar", limits=c(cmin, cmax))
ggsave(p + theme(legend.position = 'none'), 
       filename = paste0(figdir, 'LF-scatter-12-ompA-annotated.pdf'),
       width = 7, height = 7, units = 'cm')


p = plotFactorScatter2(
  model,
  factors = 1:2,
  viewidx = 1,
  color_by = 'ompA',
  showMissing = F
) +scale_color_gradientn(colours = c('#4878d0', 'white', '#ee854a'), 
                         values = scales::rescale(c(cmin,0,cmax)),
                         guide = "colorbar", limits=c(cmin, cmax)) +
  guides(colour = guide_colourbar(title.position="top", title.hjust = 0.5))
ggsave(p + theme(legend.position = 'none'), 
       filename = paste0(figdir, 'LF-scatter-12-ompA.pdf'),
       width = 5, height = 5, units = 'cm')

library(ggpubr)
leg = as_ggplot(get_legend(p + theme(legend.position = 'top',
                                     legend.key.height = unit(0.25, 'cm'))))
ggsave(leg, filename = paste0(figdir, 'ompA-legend.pdf'),
       height=2, width = 4, units = 'cm')

# for LF2 'purK' has high loadings
cmin = min(TrainData(model)[[1]]['purK',], na.rm=T)
cmax = max(TrainData(model)[[1]]['purK',], na.rm=T)
p = plotFactorScatter2(
  model,
  factors = 1:2,
  viewidx = 1,
  color_by = 'purK',
  showMissing = F
) +   ggrepel::geom_text_repel(aes(label = ifelse((y > 0.5 & color_by < -2) | 
                                                    (color_by > 1 & y < 0), drug, ""),
                                   color = color_by),
                               size = 7*5/14) +
   scale_color_gradientn(colours = c('#4878d0', 'white', '#ee854a'), 
                        values = scales::rescale(c(cmin,0,cmax)),
                        guide = "colorbar", limits=c(cmin, cmax))
ggsave(p + theme(legend.position = 'none'), 
       filename = paste0(figdir, 'LF-scatter-12-purK-annotated.pdf'),
       width = 7, height = 7, units = 'cm')


p = plotFactorScatter2(
  model,
  factors = 1:2,
  viewidx = 1,
  color_by = 'purK',
  showMissing = F
) +  scale_color_gradientn(colours = c('#4878d0', 'white', '#ee854a'), 
                           values = scales::rescale(c(cmin,0,cmax)),
                           guide = "colorbar", limits=c(cmin, cmax)) + 
  guides(colour = guide_colourbar(title.position="top", title.hjust = 0.5))
ggsave(p + theme(legend.position = 'none'), 
       filename = paste0(figdir, 'LF-scatter-12-purK.pdf'),
       width = 5, height = 5, units = 'cm')

leg = as_ggplot(get_legend(p + theme(legend.position = 'top',
                                     legend.key.height = unit(0.25, 'cm'))))
ggsave(leg, filename = paste0(figdir, 'purK-legend.pdf'),
       height=2, width = 4, units = 'cm')


cmin = min(TrainData(model)[[1]]['macB',], na.rm=T)
cmax = max(TrainData(model)[[1]]['macB',], na.rm=T)

p = plotFactorScatter2(
  model,
  factors = 3:4,
  viewidx = 1,
  color_by = 'macB',
  showMissing = F
) +   ggrepel::geom_text_repel(aes(label = ifelse(( y < -1) | (y > 2), drug, ""),
                                   color = color_by), size = 5*7/14) +
  scale_color_gradientn(colours = c('#4878d0', 'white', '#ee854a'), 
                        values = scales::rescale(c(cmin,0,cmax)),
                        guide = "colorbar", limits=c(cmin, cmax))
ggsave(p + theme(legend.position = 'none'), 
       filename = paste0(figdir, 'LF-scatter-34-macB-annotated.pdf'),
       width = 7, height = 7, units = 'cm')


p = plotFactorScatter2(
  model,
  factors = 3:4,
  viewidx = 1,
  color_by = 'macB',
  showMissing = F
) +  scale_color_gradientn(colours = c('#4878d0', 'white', '#ee854a'), 
                        values = scales::rescale(c(cmin,0,cmax)),
                        guide = "colorbar", limits=c(cmin, cmax)) +
  guides(colour = guide_colourbar(title.position="top", title.hjust = 0.5))
ggsave(p + theme(legend.position = 'none'), 
       filename = paste0(figdir, 'LF-scatter-34-macB.pdf'),
       width = 5, height = 5, units = 'cm')

leg = as_ggplot(get_legend(p + theme(legend.position = 'top',
                                     legend.key.height = unit(0.25, 'cm'))))
ggsave(leg, filename = paste0(figdir, 'macB-legend.pdf'),
       height=2, width = 4, units = 'cm')


cmin = min(TrainData(model)[[1]]['wzyE',], na.rm=T)
cmax = max(TrainData(model)[[1]]['wzyE',], na.rm=T)

p = plotFactorScatter2(
  model,
  factors = 3:4,
  viewidx = 1,
  color_by = 'wzyE',
  showMissing = F
) +   ggrepel::geom_text_repel(aes(label = ifelse(( x < -0.5 & color_by < 0) | (x > 1 & color_by > 0), drug, ""),
                                   color = color_by), size = 5*7/14) +
  scale_color_gradientn(colours = c('#4878d0', 'white', '#ee854a'), 
                        values = scales::rescale(c(cmin,0,cmax)),
                        guide = "colorbar", limits=c(cmin, cmax))
ggsave(p + theme(legend.position = 'none'), 
       filename = paste0(figdir, 'LF-scatter-34-wzyE-annotated.pdf'),
       width = 7, height = 7, units = 'cm')


p = plotFactorScatter2(
  model,
  factors = 3:4,
  viewidx = 1,
  color_by = 'wzyE',
  showMissing = F
) +  scale_color_gradientn(colours = c('#4878d0', 'white', '#ee854a'), 
                           values = scales::rescale(c(cmin,0,cmax)),
                           guide = "colorbar", limits=c(cmin, cmax)) +
  guides(colour = guide_colourbar(title.position="top", title.hjust = 0.5))
ggsave(p + theme(legend.position = 'none'), 
       filename = paste0(figdir, 'LF-scatter-34-wzyE.pdf'),
       width = 5, height = 5, units = 'cm')

leg = as_ggplot(get_legend(p + theme(legend.position = 'top',
                                     legend.key.height = unit(0.25, 'cm'))))
ggsave(leg, filename = paste0(figdir, 'wzyE-legend.pdf'),
       height=2, width = 4, units = 'cm')



# plot top weights for LF1
plotTopWeights(
  model, 
  view="EC chemical genetics", 
  factor=1
)

plotTopWeights(
  model, 
  view="ST chemical genetics", 
  factor=1
)

plotTopWeights(
  model, 
  view="EC drug interactions", 
  factor=1
)

plotTopWeights(
  model, 
  view="ST drug interactions", 
  factor=1
)


# plot top weights for LF2
plotTopWeights(
  model, 
  view="EC chemical genetics", 
  factor=2
)

plotTopWeights(
  model, 
  view="ST chemical genetics", 
  factor=2
)
