require(magrittr)
require(dplyr)
require(stringi)

datapath <- "raw_data"

articles <- readRDS(datapath)

#assume you've collected data.frame with some title, body and abstract

split_by_sentence <- function (text) {
  
  # split based on periods, exclams or question marks
  result <- unlist (strsplit (text, split = "[\\.!?]+"))
  
  # do not return empty strings
  result <- stri_trim_both (result)
  result <- result [nchar (result) > 0]
  
  # ensure that something is always returned
  if (length (result) == 0)
    result <- ""
  
  return (result)
}

place_tags <- function(x, type = "abstract"){
  x %>% split_by_sentence %>% paste0(collapse = ".  <s> </s> ") %>% paste0(type, "=<d> <p> <s> ", . , ". </s> <p> </d>")
}

articles$data <- paste(articles$body %>% sapply(place_tags, type = "article"), articles$abstract %>% sapply(place_tags))

articles$abstract %<>% place_tags
articles$body %<>% place_tags(type = "article")

for(i in 1:nrow(articles)){
  write(articles$data[i], paste0("data_", i, ".txt"))
}
