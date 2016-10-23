require(rplos)
require(dplyr)

res <- searchplos(
  q = '*:*', limit = 20000, #q = 'medicine'
  fl = plosfields$field,
  fq = list('doc_type:full', '-article_type:correction','-article_type:viewpoints')
)
saveRDS(res$data %<>% filter(!is.na(abstract) & abstract != ""), "Data/no_spec_topic_articles")
rm(res)
gc()
