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
