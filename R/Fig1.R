## Plots for figure 1
library(dplyr)
library(ggplot2)

# aggregate chemgen data across concentrations
agg_conc <- function(df, fdr = 0.05) {
  df_avg = group_by(df, gene, drug) %>%
    summarise(hit = sum(padj < fdr),
              sscore = sscore[which.max(abs(sscore))]) %>%
    ungroup()
  df_avg
}

basedir = "~/Documents/github/chemgen"
datadir = file.path(basedir, 'data/shiny')
figdir = file.path(basedir, "figures/")
# color palette
pal = c("#FFCC33", "#009999")


ecoli_chemgen = agg_conc(readRDS(file.path(datadir, 'nichols.rds')))
st_chemgen = agg_conc(readRDS(file.path(datadir, 'ST_chemgen.rds')))

# map E. coli gene names
string.map = read.csv(file.path(datadir, "string_mapping-EcoliBW.tsv"),
                      header = T,
                      sep = "\t", stringsAsFactors = F)
ecoli_chemgen = mutate(ecoli_chemgen, 
                    gene = plyr::mapvalues(gene, from = string.map$queryItem,
                                           to = string.map$preferredName))

# get drug-gene interactions
get_drug_gene <- function(df) {
  filter(df, hit > 0) %>%
    mutate(sign = ifelse(sscore >= 0, "antagonism", "synergy")) %>%
    distinct(drug, gene, sign)
}

ecoli_chemgen = get_drug_gene(ecoli_chemgen)
ecoli_chemgen$species = 'E. coli'

st_chemgen = get_drug_gene(st_chemgen)
st_chemgen$species = 'S. typhimurium'

chemgen_counts = bind_rows(ecoli_chemgen, st_chemgen)
p = ggplot(chemgen_counts, aes(x=species, fill = sign) )+
  geom_bar(position = 'dodge', width = 0.7) + 
  scale_fill_manual(values = pal)+
  labs(x="", y="", fill="", title = "Drug-gene interactions")+
  coord_flip() +
  scale_x_discrete(expand = c(0,0))+
  scale_y_continuous(expand=c(0,0))+
  theme_classic(base_size = 9)+
  theme(axis.ticks.y = element_blank(),
        axis.line.y = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 9))
ggsave(p + theme(legend.position = 'none'), 
       filename = paste0(figdir, 'Chemgen-barplot-species.pdf'),
       width = 5.5, height = 2.5, units = 'cm')

# plot top genes ranked by number of drug-gene interactions
df = group_by(chemgen_counts, species, gene, sign) %>%
  summarise(n=n()) 

top_genes = group_by(df, gene) %>%
  summarise(n = sum(n)) %>%
  slice_max(n = 20, order_by = n)

p = ggplot(filter(df, gene %in% top_genes$gene) %>%
         mutate(Strain = plyr::mapvalues(species,
                                         from = c("E. coli",
                                                  "S. typhimurium"),
                                         to = c("EC", "ST"))),
       aes(x= Strain, 
           y = factor(gene, levels = rev(top_genes$gene)), 
           color = sign)) +
  geom_point(aes(size = n)) +
  facet_wrap(~ sign) + 
  scale_color_manual(values = pal)+
  labs(x = "", y = "",
       title = '') + 
  theme_classic(base_size = 9) +
  theme(axis.line = element_blank(),
        #axis.text.x = element_blank(),
        axis.ticks = element_blank(),
        strip.background = element_blank(),
        strip.text = element_blank(),
        plot.title = element_text(size = 9,
                                  hjust = 0.5,
                                  face = 'italic'))
ggsave(p+theme(legend.position = 'none'),
       filename = paste0(figdir, "topgenes-chemgen-dotplot-top20.pdf"), 
       width = 5, height =15, units = 'cm')


sel_genes = c("acrB", "hfq", "asmA", "tolQ", "tatC",
              "fis", "cysB", "purA")

ord = filter(df, gene %in% sel_genes) %>%
  group_by(gene) %>%
  summarise(n = sum(n)) %>%
  arrange(n)

p = ggplot(filter(df, gene %in% sel_genes) %>%
         mutate(Strain = plyr::mapvalues(species,
                                         from = c("E. coli",
                                                  "S. typhimurium"),
                                         to = c("EC", "ST"))),
       aes(x= Strain, 
           y = factor(gene, levels = ord$gene), 
           color = sign)) +
  geom_point(aes(size = n)) +
  facet_wrap(~ sign) + 
  scale_color_manual(values = pal)+
  labs(x = "", y = "",
       title = '') + 
  theme_classic(base_size = 9) +
  theme(axis.line = element_blank(),
        #axis.text.x = element_blank(),
        axis.ticks = element_blank(),
        strip.background = element_blank(),
        strip.text = element_blank(),
        plot.title = element_text(size = 9,
                                  hjust = 0.5,
                                  face = 'italic'))
ggsave(p+theme(legend.position = 'none'),
       filename = paste0(figdir, "topgenes-chemgen-dotplot.pdf"), 
       width = 5, height =6, units = 'cm')

# E. coli drug-drug interaction data
ecoli_df = readRDS(file.path(datadir, 'EcoliBW_refset.rds'))
ST_df = readRDS(file.path(datadir, 'ST14028_refset.rds'))

ECST = bind_rows(mutate(ecoli_df, species = 'E. coli'),
                 mutate(ST_df, species = 'S. typhimurium') )

p = ggplot(filter(ECST, type != 'none'), aes(x=species, fill = type) )+
  geom_bar(position = 'dodge', width = 0.7) + 
  scale_fill_manual(values = pal)+
  labs(x="", y = "", title="Drug-drug interactions", fill="")+
  coord_flip() +
  scale_x_discrete(expand = c(0,0))+
  scale_y_continuous(expand=c(0,0))+
  theme_classic(base_size = 9)+
  theme(axis.ticks.y = element_blank(),
        axis.line.y = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 9))
ggsave(p + theme(legend.position = 'none'), 
       filename = paste0(figdir, 'Bliss-barplot-species.pdf'),
       width = 5.5, height = 2.5, units = 'cm')


# add the drug class information to check what are 
# the top interacting drug classes
drugleg = read.delim2(file.path(basedir, "data/chemicals/legend_gramnegpos.txt"))
drugleg = select(drugleg, Drug, Class_refined)
drugleg = mutate(drugleg, Class_refined = plyr::mapvalues(Class_refined,
                                        from = c('phosphoenolpyruvate_analogue',
                                                 'cephalosporin',
                                                 'penicillin'),
                                        to = c('Fosfomycin',
                                               'beta-lactam',
                                               'beta-lactam')))

ECST = inner_join(ECST, rename(drugleg,
                            d1 = Drug,
                            class1 = Class_refined))
ECST = inner_join(ECST, rename(drugleg,
                                       d2 = Drug,
                                       class2 = Class_refined))
ECST = filter(ECST, type != 'none')

df = bind_rows(select(ECST, type, class1) %>%
            rename(class = class1),
          select(ECST, type, class2) %>%
            rename(class = class2))

df = group_by(df, Strain, type, class) %>%
  summarise(n=n()) %>%
  group_by(Strain, type) %>%
  slice_max(n = 5, order_by = n) %>%
  mutate(rank = row_number())

p = ggplot(filter(df, class %in% c("aminoglycoside",
                               "beta-lactam",
                               "macrolide",
                               "phenazine")) %>%
             mutate(Strain = plyr::mapvalues(Strain,
                                             from = c("E. coli BW",
                                                      "ST 14028s"),
                                             to = c("EC", "ST"))),
           aes(x= Strain, 
               y = factor(class, levels = c('phenazine',
                                            "macrolide", 
                                            "aminoglycoside",
                                            'beta-lactam')), 
               color = type)) +
  geom_point(aes(size = n)) +
  facet_wrap(~ type) + 
  scale_color_manual(values = pal)+
  labs(x = "", y = "",
       title = '') + 
  theme_classic(base_size = 9) +
  theme(axis.line = element_blank(),
        #axis.text.x = element_blank(),
        axis.ticks = element_blank(),
        strip.background = element_blank(),
        strip.text = element_blank(),
        plot.title = element_text(size = 9,
                                  hjust = 0.5,
                                  face = 'italic'))
ggsave(p+theme(legend.position = 'none'),
       filename = paste0(figdir, "topdrugclass-druginter.pdf"), 
       width = 5, height =5, units = 'cm')
