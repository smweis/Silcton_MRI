library(readxl)
myData <- read_excel("DataAnalysisWith70Participants_Jupyter.xlsx")


fit <- lm(Pointing_Total ~ Right_Head, data=myData)
summary(fit) # show results
