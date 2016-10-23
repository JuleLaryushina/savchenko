require(magrittr)
require(dplyr)

datapath <- "raw_data"

articles <- readRDS(datapath)

articles$abstract[1]

require(quanteda)

df_m <- corpus(c(articles$body, articles$abstract)) %>% 
  dfm(removePunct = FALSE,
      removeNumbers = FALSE)
df <- data.frame(
  word = df_m@Dimnames$features,
  counts = colSums(df_m)
)
df %<>% arrange(desc(counts))
df %<>% filter(counts >= 55)
write.table(df, file = "my_vocab", row.names = FALSE, col.names = FALSE, quote = FALSE)
