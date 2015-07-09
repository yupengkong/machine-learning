library(rpart)
library(tree)
library(rattle)

# 199523 observations
# 187141 observations '- 50000.'
# 12382 observations '50000+.

# decision tree
dat = read.table('census-income.data', header = F, sep = ',')
target <- 'V42'
nobs <- nrow(dat)
form <- formula(paste(target, "~ ."))

#form2 <- formula(paste(target, "~ V10 + V5 + V9 + V3 + V40 + V17 + V2 + V13 + V4 + V19 + V1 + V23 + V24"))
sample_size <- floor(0.8 * nobs)
set.seed(42)
space <- sample(seq_len(nobs), size = sample_size)
train <- dat[space, ]
test <- dat[-space, ]
model <- rpart(form, train, method = 'class', parms=list(split="information"))
preds <- predict(model, test, type = 'class')
class.pred <- table(preds, test[[target]])
errate1 = 1-sum(diag(class.pred))/sum(class.pred)

# using V10, V5, V17, V13, V1 and V19(error rate = 0.05242)


# randomForest
library(randomForest)
dat = read.table('census-income.data', header = F, sep = ',')
target <- 'V42'
nobs <- nrow(dat)
form <- formula(paste(target, "~ ."))
sample_size <- floor(0.8 * nobs)
set.seed(42)
space <- sample(seq_len(nobs), size = sample_size)
train <- dat[space, ]
test <- dat[-space, ]
model <- randomForest(formula=form,
                      data=train, mtry=4,
                      importance=TRUE,
                      keep.forest=TRUE,
                      na.action=na.roughfix,
                      replace=FALSE)



colnames(dat)[1] = 'age'
colnames(dat)[2] = 'workerclass'             #9
colnames(dat)[3] = 'detail_industry'         #52
colnames(dat)[4] = 'detail_occupation'       #47
colnames(dat)[5] = 'education'               #17
colnames(dat)[6] = 'wage'                     
colnames(dat)[7] = 'eduenroll'               #3
colnames(dat)[8] = 'marital'                 #7
colnames(dat)[9] = 'major_industry'          #24
colnames(dat)[10] = 'major_occupation'       #15
colnames(dat)[11] = 'race'                   #5
colnames(dat)[12] = 'hispan_origin'  # NA in this column   #10
colnames(dat)[13] = 'sex'                   #2
colnames(dat)[14] = 'labor_union'           #3
colnames(dat)[15] = 'losejobreason'         #6
colnames(dat)[16] = 'full_part'             #8
colnames(dat)[17] = 'gain'                  
colnames(dat)[18] = 'loss'
colnames(dat)[19] = 'dividends'
colnames(dat)[20] = 'taxfiler'              #6
colnames(dat)[21] = 'region'                #6
colnames(dat)[22] = 'state'      # ? in this column
colnames(dat)[23] = 'family'                #38 in train, 37 in test
colnames(dat)[24] = 'household'             #8
colnames(dat)[25] = 'weight'
colnames(dat)[26] = 'migration_msa'         # ? in this column
colnames(dat)[27] = 'migrationchange_reg'   # ? in this column
colnames(dat)[28] = 'migrationmove_reg'     # ? in this column
colnames(dat)[29] = 'liveduration'          #3
colnames(dat)[30] = 'migrationprev'         # ? in this column
colnames(dat)[31] = 'numpersonworking'      #7
colnames(dat)[32] = 'familymembersunder18'  #5
colnames(dat)[33] = 'fatherborn'            # ? in this column
colnames(dat)[34] = 'motherborn'            # ? in this column
colnames(dat)[35] = 'selfborn'              # ? in this column
colnames(dat)[36] = 'citizen'               #5
colnames(dat)[37] = 'ownbusiness'           #3
colnames(dat)[38] = 'veternadmin'           #3
colnames(dat)[39] = 'veternbenefit'         #3 
colnames(dat)[40] = 'workweek'
colnames(dat)[41] = 'year'                  #deleted
colnames(dat)[42] = 'target'

