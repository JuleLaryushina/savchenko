# Copyright 2016 Julia Laryushina. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
